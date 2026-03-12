"""
Reproducible 3-method comparison with multiple seeds.
"""

import sys
import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage import distance_transform_edt, binary_dilation
import json
import random

torch.set_num_threads(32)

# =============================================================================
# SETTINGS
# =============================================================================

GZ21_PATH = '.'
ORIGINAL_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc'
FILLED_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/Neumann_DrichletBc/cplusplus/fordays/sst_filled_365days_2023.nc'
ORIGINAL_VAR_NAME = 'analysed_sst'
FILLED_VAR_NAME = 'sst_neumann'
FILL_VALUE = -999.0

COARSEN_FACTOR = 4
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
TRAIN_FRACTION = 0.8
MAX_TIME_STEPS = 364

OUTPUT_DIR = './output_three_methods_multiseed'

# Multiple seeds for statistical robustness
SEEDS = [42, 123, 456, 789, 1011]

# =============================================================================
# Seed function
# =============================================================================

def set_all_seeds(seed):
    """Fix ALL sources of randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  All seeds set to {seed}")

# =============================================================================
# Import GZ21 model
# =============================================================================

sys.path.insert(0, GZ21_PATH)
from models.models1 import FullyCNN

# =============================================================================
# Helper classes (same as before)
# =============================================================================

class SoftPlusTransform(nn.Module):
    def __init__(self, n_targets=1):
        super().__init__()
        self.n_targets = n_targets

    def forward(self, x):
        mean = x[:, :self.n_targets, :, :]
        log_std = x[:, self.n_targets:, :, :]
        std = F.softplus(log_std) + 1e-6
        return torch.cat([mean, std], dim=1)


class HeteroskedasticGaussianLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        mean = output[:, 0:1, :, :]
        std = output[:, 1:2, :, :]
        variance = std ** 2
        nll = 0.5 * torch.log(variance) + 0.5 * ((target - mean) ** 2) / variance
        if mask is not None:
            nll = nll * mask
            return nll.sum() / (mask.sum() + 1e-8)
        return nll.mean()


class TemperatureDataset(Dataset):
    def __init__(self, T_input, S_T_target, ocean_mask):
        self.T = torch.tensor(T_input, dtype=torch.float32)
        self.S_T = torch.tensor(S_T_target, dtype=torch.float32)
        self.mask = torch.tensor(ocean_mask, dtype=torch.float32)

        valid = ocean_mask.astype(bool)
        self.T_mean = np.nanmean(T_input[:, valid])
        self.T_std = np.nanstd(T_input[:, valid]) + 1e-8
        self.S_mean = np.nanmean(S_T_target[:, valid])
        self.S_std = np.nanstd(S_T_target[:, valid]) + 1e-8

        self.T = (self.T - self.T_mean) / self.T_std
        self.S_T = (self.S_T - self.S_mean) / self.S_std
        self.T = torch.nan_to_num(self.T, nan=0.0)
        self.S_T = torch.nan_to_num(self.S_T, nan=0.0)

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, idx):
        return self.T[idx].unsqueeze(0), self.S_T[idx].unsqueeze(0)

# =============================================================================
# Functions
# =============================================================================

def compute_subgrid_forcing(T_highres, factor, chunk_size=50):
    """Memory-efficient coarsening."""
    n_times, ny, nx = T_highres.shape
    ny_trim = (ny // factor) * factor
    nx_trim = (nx // factor) * factor
    ny_coarse = ny_trim // factor
    nx_coarse = nx_trim // factor

    T_coarse = np.zeros((n_times, ny_coarse, nx_coarse), dtype=np.float32)
    S_T = np.zeros((n_times, ny_coarse, nx_coarse), dtype=np.float32)

    for t_start in range(0, n_times, chunk_size):
        t_end = min(t_start + chunk_size, n_times)
        T_chunk = T_highres[t_start:t_end, :ny_trim, :nx_trim].astype(np.float32)
        nt_chunk = t_end - t_start
        T_reshaped = T_chunk.reshape(nt_chunk, ny_coarse, factor, nx_coarse, factor)
        T_coarse[t_start:t_end] = np.nanmean(T_reshaped, axis=(2, 4))
        T_sq_coarse = np.nanmean(T_reshaped ** 2, axis=(2, 4))
        S_T[t_start:t_end] = T_sq_coarse - T_coarse[t_start:t_end] ** 2
        del T_chunk, T_reshaped, T_sq_coarse

    return T_coarse, S_T


def create_replicate_fill(T_with_nan, land_mask_3d):
    """Replicate fill: nearest neighbor."""
    n_times, ny, nx = T_with_nan.shape
    T_replicate = T_with_nan.astype(np.float32).copy()
    land_mask_2d = land_mask_3d[0]
    _, indices = distance_transform_edt(land_mask_2d, return_indices=True)
    land_rows = indices[0][land_mask_2d]
    land_cols = indices[1][land_mask_2d]

    for t in range(n_times):
        T_replicate[t, land_mask_2d] = T_with_nan[t, land_rows, land_cols]

    T_replicate = np.nan_to_num(T_replicate, nan=288.0)
    return T_replicate


def train_model(train_loader, val_loader, mask_tensor, device, method_name):
    """Train model."""
    model = FullyCNN(n_in_channels=1, n_out_channels=2, padding='same')
    model.final_transformation = SoftPlusTransform(n_targets=1)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = HeteroskedasticGaussianLoss()

    best_loss = float('inf')
    best_state = None
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y, mask_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y, mask_tensor)
                val_loss += loss.item() * x.size(0)
        val_loss /= val_size

        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, data_loader, ocean_mask, coastal_mask, S_mean, S_std, device):
    """Evaluate and compute R²."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out[:, 0:1, :, :].cpu().numpy() * S_std + S_mean
            target = y.cpu().numpy() * S_std + S_mean
            all_preds.append(pred)
            all_targets.append(target)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    def compute_r2(mask):
        p = preds[:, 0, mask].flatten()
        t = targets[:, 0, mask].flatten()
        valid = np.isfinite(p) & np.isfinite(t)
        p, t = p[valid], t[valid]
        if len(p) == 0:
            return np.nan
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)

    open_ocean_mask = ocean_mask & ~coastal_mask
    return {
        'r2_overall': compute_r2(ocean_mask),
        'r2_coastal': compute_r2(coastal_mask),
        'r2_open_ocean': compute_r2(open_ocean_mask),
    }


def main():
    print("=" * 60)
    print("MULTI-SEED COMPARISON")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Load and prepare data (ONCE, outside seed loop)
    # =========================================================================
    print("\nLoading data...")
    ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
    T_orig = ds_orig[ORIGINAL_VAR_NAME].values[:MAX_TIME_STEPS, :, :].astype(np.float32)
    ds_orig.close()

    land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
    T_orig_with_nan = T_orig.copy()
    T_orig_with_nan[land_mask] = np.nan

    # Three fill methods
    print("Creating filled versions...")
    T_zero = np.nan_to_num(T_orig_with_nan, nan=0.0).astype(np.float32)
    T_replicate = create_replicate_fill(T_orig_with_nan, land_mask)
    
    ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
    T_laplace = ds_filled[FILLED_VAR_NAME].values[:MAX_TIME_STEPS, 0, :, :].astype(np.float32)
    ds_filled.close()

    # Coarsen
    print("Coarsening...")
    T_coarse_orig, S_T_target = compute_subgrid_forcing(T_orig_with_nan, COARSEN_FACTOR)
    ocean_mask = np.isfinite(T_coarse_orig[0])
    land_mask_coarse = ~ocean_mask
    coastal_mask = binary_dilation(land_mask_coarse, iterations=10) & ocean_mask

    S_T_target_masked = S_T_target.copy()
    S_T_target_masked[:, ~ocean_mask] = 0.0

    T_coarse_zero, _ = compute_subgrid_forcing(T_zero, COARSEN_FACTOR)
    T_coarse_replicate, _ = compute_subgrid_forcing(T_replicate, COARSEN_FACTOR)
    T_coarse_laplace, _ = compute_subgrid_forcing(T_laplace, COARSEN_FACTOR)

    # Free memory
    del T_orig, T_orig_with_nan, T_zero, T_replicate, T_laplace

    # Mask tensor
    mask_tensor = torch.tensor(ocean_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # =========================================================================
    # Run for each seed
    # =========================================================================
    all_results = {
        'zero_fill': {'overall': [], 'coastal': [], 'open_ocean': []},
        'replicate_fill': {'overall': [], 'coastal': [], 'open_ocean': []},
        'laplace_fill': {'overall': [], 'coastal': [], 'open_ocean': []},
    }

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED = {seed}")
        print(f"{'='*60}")

        # Set all seeds
        set_all_seeds(seed)

        # Create datasets
        dataset_zero = TemperatureDataset(T_coarse_zero, S_T_target_masked, ocean_mask)
        dataset_replicate = TemperatureDataset(T_coarse_replicate, S_T_target_masked, ocean_mask)
        dataset_laplace = TemperatureDataset(T_coarse_laplace, S_T_target_masked, ocean_mask)

        n = len(dataset_zero)
        train_size = int(TRAIN_FRACTION * n)
        val_size = n - train_size

        # Split with same seed
        gen = torch.Generator().manual_seed(seed)
        train_zero, val_zero = random_split(dataset_zero, [train_size, val_size], generator=gen)
        gen = torch.Generator().manual_seed(seed)
        train_replicate, val_replicate = random_split(dataset_replicate, [train_size, val_size], generator=gen)
        gen = torch.Generator().manual_seed(seed)
        train_laplace, val_laplace = random_split(dataset_laplace, [train_size, val_size], generator=gen)

        # Data loaders
        train_loader_zero = DataLoader(train_zero, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader_zero = DataLoader(val_zero, batch_size=BATCH_SIZE, num_workers=0)
        train_loader_replicate = DataLoader(train_replicate, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader_replicate = DataLoader(val_replicate, batch_size=BATCH_SIZE, num_workers=0)
        train_loader_laplace = DataLoader(train_laplace, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader_laplace = DataLoader(val_laplace, batch_size=BATCH_SIZE, num_workers=0)

        # Train and evaluate
        for name, train_loader, val_loader, dataset in [
            ('zero_fill', train_loader_zero, val_loader_zero, dataset_zero),
            ('replicate_fill', train_loader_replicate, val_loader_replicate, dataset_replicate),
            ('laplace_fill', train_loader_laplace, val_loader_laplace, dataset_laplace),
        ]:
            print(f"\n  Training {name}...")
            model = train_model(train_loader, val_loader, mask_tensor, device, name)
            results = evaluate_model(model, val_loader, ocean_mask, coastal_mask,
                                     dataset.S_mean, dataset.S_std, device)

            all_results[name]['overall'].append(results['r2_overall'])
            all_results[name]['coastal'].append(results['r2_coastal'])
            all_results[name]['open_ocean'].append(results['r2_open_ocean'])

            print(f"    R² = {results['r2_overall']:.4f}")

    # =========================================================================
    # Compute statistics
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL RESULTS (mean ± std over 5 seeds)")
    print(f"{'='*60}")

    final_results = {}
    for method in ['zero_fill', 'replicate_fill', 'laplace_fill']:
        final_results[method] = {}
        for region in ['overall', 'coastal', 'open_ocean']:
            values = all_results[method][region]
            mean = np.mean(values)
            std = np.std(values)
            final_results[method][region] = {'mean': mean, 'std': std, 'values': values}

    # Print table
    print(f"\n{'Method':<18} {'Overall R²':>18} {'Coastal R²':>18} {'Open Ocean R²':>18}")
    print("-" * 76)
    for method in ['zero_fill', 'replicate_fill', 'laplace_fill']:
        o = final_results[method]['overall']
        c = final_results[method]['coastal']
        oo = final_results[method]['open_ocean']
        print(f"{method:<18} {o['mean']:.3f} ± {o['std']:.3f}      "
              f"{c['mean']:.3f} ± {c['std']:.3f}      "
              f"{oo['mean']:.3f} ± {oo['std']:.3f}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'results_multiseed.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n✓ Results saved to {OUTPUT_DIR}/results_multiseed.json")

    # =========================================================================
    # Plot with error bars
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
    x = np.arange(len(methods))
    width = 0.25

    for i, (region, color) in enumerate([('overall', 'steelblue'), 
                                          ('coastal', 'coral'), 
                                          ('open_ocean', 'seagreen')]):
        means = [final_results[m.lower().replace('-', '_')][region]['mean'] for m in methods]
        stds = [final_results[m.lower().replace('-', '_')][region]['std'] for m in methods]
        
        bars = ax.bar(x + (i - 1) * width, means, width, 
                      yerr=stds, capsize=5,
                      label=region.replace('_', ' ').title(), 
                      color=color, alpha=0.8)

    ax.set_ylabel('R² Score')
    ax.set_title('CNN Performance: Three Land Fill Methods\n(Mean ± Std over 5 seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods_errorbars.png'), dpi=150)
    plt.close()

    print("✓ Plot saved")
    print("\nDone!")

    # =========================================================================
    # Professional Color Palette
    # =========================================================================
    COLORS = ['#3498DB', '#E67E22', '#2ECC71']  # Sky Blue, Carrot, Emerald
    
    # =========================================================================
    # Plot 1: With Error Bars (Multi-seed)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    
    methods = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
    regions = ['overall', 'coastal', 'open_ocean']
    region_labels = ['Overall', 'Coastal', 'Open Ocean']
    x = np.arange(len(methods))
    width = 0.25
    
    for i, (region, label, color) in enumerate(zip(regions, region_labels, COLORS)):
        means = [final_results[m.lower().replace('-', '_')][region]['mean'] for m in methods]
        stds = [final_results[m.lower().replace('-', '_')][region]['std'] for m in methods]
        
        bars = ax.bar(x + (i - 1) * width, means, width,
                      yerr=stds, capsize=4,
                      label=label,
                      color=color, 
                      edgecolor='white', 
                      linewidth=0.7,
                      error_kw={'linewidth': 1.2, 'capthick': 1.2})
        
        # Add mean values on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height + stds[bars.index(bar)] + 0.01),
                       ha='center', va='bottom', fontsize=8, fontweight='medium')
    
    ax.set_ylabel('$R^2$ Score', fontsize=12)
    ax.set_xlabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, max([final_results[m.lower().replace('-', '_')]['open_ocean']['mean'] for m in methods]) * 1.25)
    
    # Clean style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods_errorbars.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods_errorbars.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Plot with error bars saved")
    
    # =========================================================================
    # Plot 2: Without Error Bars (Single Values)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for i, (region, label, color) in enumerate(zip(regions, region_labels, COLORS)):
        # Use mean values as single values
        values = [final_results[m.lower().replace('-', '_')][region]['mean'] for m in methods]
        
        bars = ax.bar(x + (i - 1) * width, values, width,
                      label=label,
                      color=color, 
                      edgecolor='white', 
                      linewidth=0.7)
        
        # Add values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='medium')
    
    ax.set_ylabel('$R^2$ Score', fontsize=12)
    ax.set_xlabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim(0, max([final_results[m.lower().replace('-', '_')]['open_ocean']['mean'] for m in methods]) * 1.2)
    
    # Clean style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Plot without error bars saved")
    
    print("\nDone!")
    
    

if __name__ == '__main__':
    main()
