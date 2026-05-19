"""
MPI-Parallel 3-method comparison: FIXED VERSION
- Only Rank 0 loads data
- Broadcasts to other ranks
- Each rank handles one seed

Run with: ibrun python threemethods_mpi_fixed.py
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

# MPI
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

torch.set_num_threads(8)  # 8 threads per rank

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

OUTPUT_DIR = './output_mpi_parallel'

# ALL SEEDS
ALL_SEEDS = [42, 123, 456, 789, 1011, 12, 124, 458, 8, 65, 788, 25, 9, 1120]

# =============================================================================
# Seed function
# =============================================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# Import GZ21 model
# =============================================================================

sys.path.insert(0, GZ21_PATH)
from models.models1 import FullyCNN

# =============================================================================
# Helper classes
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


def train_model(train_loader, val_loader, mask_tensor, device):
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


def run_single_seed(seed, T_coarse_zero, T_coarse_replicate, T_coarse_laplace,
                    S_T_target_masked, ocean_mask, coastal_mask, mask_tensor, device):
    
    set_all_seeds(seed)
    
    dataset_zero = TemperatureDataset(T_coarse_zero, S_T_target_masked, ocean_mask)
    dataset_replicate = TemperatureDataset(T_coarse_replicate, S_T_target_masked, ocean_mask)
    dataset_laplace = TemperatureDataset(T_coarse_laplace, S_T_target_masked, ocean_mask)

    n = len(dataset_zero)
    train_size = int(TRAIN_FRACTION * n)
    val_size = n - train_size

    results = {}
    
    for name, dataset in [
        ('zero_fill', dataset_zero),
        ('replicate_fill', dataset_replicate),
        ('laplace_fill', dataset_laplace),
    ]:
        gen = torch.Generator().manual_seed(seed)
        train_data, val_data = random_split(dataset, [train_size, val_size], generator=gen)
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
        
        model = train_model(train_loader, val_loader, mask_tensor, device)
        
        eval_results = evaluate_model(model, val_loader, ocean_mask, coastal_mask,
                                      dataset.S_mean, dataset.S_std, device)
        
        results[name] = {
            'overall': eval_results['r2_overall'],
            'coastal': eval_results['r2_coastal'],
            'open_ocean': eval_results['r2_open_ocean'],
        }
    
    return results


def main():
    # =========================================================================
    # Distribute seeds
    # =========================================================================
    seeds_per_rank = len(ALL_SEEDS) // size
    remainder = len(ALL_SEEDS) % size
    
    if rank < remainder:
        my_seeds = ALL_SEEDS[rank * (seeds_per_rank + 1): (rank + 1) * (seeds_per_rank + 1)]
    else:
        start = remainder * (seeds_per_rank + 1) + (rank - remainder) * seeds_per_rank
        my_seeds = ALL_SEEDS[start: start + seeds_per_rank]
    
    if rank == 0:
        print("=" * 60)
        print(f"MPI PARALLEL CNN TRAINING (FIXED)")
        print(f"Total ranks: {size}")
        print(f"Total seeds: {len(ALL_SEEDS)}")
        print("=" * 60)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[Rank {rank}] Handling seeds: {my_seeds}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # =========================================================================
    # ONLY RANK 0 LOADS DATA, THEN BROADCASTS
    # =========================================================================
    
    if rank == 0:
        print("\n[Rank 0] Loading data (other ranks waiting)...")
        
        ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
        T_orig = ds_orig[ORIGINAL_VAR_NAME].values[:MAX_TIME_STEPS, :, :].astype(np.float32)
        ds_orig.close()

        land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
        T_orig_with_nan = T_orig.copy()
        T_orig_with_nan[land_mask] = np.nan

        print("[Rank 0] Creating filled versions...")
        T_zero = np.nan_to_num(T_orig_with_nan, nan=0.0).astype(np.float32)
        T_replicate = create_replicate_fill(T_orig_with_nan, land_mask)
        
        ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
        T_laplace = ds_filled[FILLED_VAR_NAME].values[:MAX_TIME_STEPS, 0, :, :].astype(np.float32)
        ds_filled.close()

        print("[Rank 0] Coarsening...")
        T_coarse_orig, S_T_target = compute_subgrid_forcing(T_orig_with_nan, COARSEN_FACTOR)
        ocean_mask = np.isfinite(T_coarse_orig[0])
        land_mask_coarse = ~ocean_mask
        coastal_mask = binary_dilation(land_mask_coarse, iterations=10) & ocean_mask

        S_T_target_masked = S_T_target.copy()
        S_T_target_masked[:, ~ocean_mask] = 0.0

        T_coarse_zero, _ = compute_subgrid_forcing(T_zero, COARSEN_FACTOR)
        T_coarse_replicate, _ = compute_subgrid_forcing(T_replicate, COARSEN_FACTOR)
        T_coarse_laplace, _ = compute_subgrid_forcing(T_laplace, COARSEN_FACTOR)

        # Free high-res memory
        del T_orig, T_orig_with_nan, T_zero, T_replicate, T_laplace, T_coarse_orig, S_T_target
        
        print("[Rank 0] Data ready! Broadcasting to other ranks...")
        
    else:
        # Other ranks: create empty placeholders
        T_coarse_zero = None
        T_coarse_replicate = None
        T_coarse_laplace = None
        S_T_target_masked = None
        ocean_mask = None
        coastal_mask = None
    
    # =========================================================================
    # BROADCAST DATA FROM RANK 0 TO ALL RANKS
    # =========================================================================
    
    T_coarse_zero = comm.bcast(T_coarse_zero, root=0)
    T_coarse_replicate = comm.bcast(T_coarse_replicate, root=0)
    T_coarse_laplace = comm.bcast(T_coarse_laplace, root=0)
    S_T_target_masked = comm.bcast(S_T_target_masked, root=0)
    ocean_mask = comm.bcast(ocean_mask, root=0)
    coastal_mask = comm.bcast(coastal_mask, root=0)
    
    if rank == 0:
        print("[Rank 0] Broadcast complete!")
    
    comm.Barrier()
    print(f"[Rank {rank}] Data received, starting training...")
    
    # Mask tensor
    mask_tensor = torch.tensor(ocean_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # =========================================================================
    # Run seeds for this rank
    # =========================================================================
    my_results = {}
    
    for seed in my_seeds:
        print(f"[Rank {rank}] Training seed {seed}...")
        
        results = run_single_seed(
            seed, T_coarse_zero, T_coarse_replicate, T_coarse_laplace,
            S_T_target_masked, ocean_mask, coastal_mask, mask_tensor, device
        )
        
        my_results[seed] = results
        
        print(f"[Rank {rank}] Seed {seed} done: "
              f"Zero={results['zero_fill']['overall']:.4f}, "
              f"Rep={results['replicate_fill']['overall']:.4f}, "
              f"Lap={results['laplace_fill']['overall']:.4f}")
    
    # =========================================================================
    # Gather results to rank 0
    # =========================================================================
    all_results = comm.gather(my_results, root=0)
    
    if rank == 0:
        # Merge all results
        merged_results = {}
        for rank_results in all_results:
            merged_results.update(rank_results)
        
        print(f"\n{'='*60}")
        print(f"ALL SEEDS COMPLETE ({len(merged_results)} seeds)")
        print(f"{'='*60}")
        
        # Compute statistics
        final_results = {
            'zero_fill': {'overall': [], 'coastal': [], 'open_ocean': []},
            'replicate_fill': {'overall': [], 'coastal': [], 'open_ocean': []},
            'laplace_fill': {'overall': [], 'coastal': [], 'open_ocean': []},
        }
        
        for seed, results in merged_results.items():
            for method in ['zero_fill', 'replicate_fill', 'laplace_fill']:
                for region in ['overall', 'coastal', 'open_ocean']:
                    final_results[method][region].append(results[method][region])
        
        # Compute stats
        stats = {}
        for method in ['zero_fill', 'replicate_fill', 'laplace_fill']:
            stats[method] = {}
            for region in ['overall', 'coastal', 'open_ocean']:
                values = final_results[method][region]
                stats[method][region] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'p25': np.percentile(values, 25),
                    'p75': np.percentile(values, 75),
                    'values': values,
                }
        
        # Print summary
        print(f"\n{'Method':<18} {'Overall R²':>18} {'Coastal R²':>18} {'Open Ocean R²':>18}")
        print("-" * 76)
        for method in ['zero_fill', 'replicate_fill', 'laplace_fill']:
            o = stats[method]['overall']
            c = stats[method]['coastal']
            oo = stats[method]['open_ocean']
            print(f"{method:<18} {o['mean']:.3f} ± {o['std']:.3f}      "
                  f"{c['mean']:.3f} ± {c['std']:.3f}      "
                  f"{oo['mean']:.3f} ± {oo['std']:.3f}")
        
        # Save results
        with open(os.path.join(OUTPUT_DIR, 'results_mpi.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else list(x) if isinstance(x, np.ndarray) else x)
        
        print(f"\n✓ Results saved to {OUTPUT_DIR}/results_mpi.json")
        
        # =====================================================================
        # Plot
        # =====================================================================
        COLORS = {
            'overall': '#0072B2',
            'coastal': '#D55E00',
            'open_ocean': '#009E73',
        }

        methods = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
        method_keys = ['zero_fill', 'replicate_fill', 'laplace_fill']
        regions = ['overall', 'coastal', 'open_ocean']
        region_labels = ['Overall', 'Coastal', 'Open Ocean']

        x = np.arange(len(methods))
        width = 0.24
        offsets = [-width, 0, width]

        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (region, label) in enumerate(zip(regions, region_labels)):
            means = [stats[m][region]['mean'] for m in method_keys]
            p25 = [stats[m][region]['p25'] for m in method_keys]
            p75 = [stats[m][region]['p75'] for m in method_keys]
            
            yerr_lower = [means[j] - p25[j] for j in range(3)]
            yerr_upper = [p75[j] - means[j] for j in range(3)]
            
            bars = ax.bar(x + offsets[i], means, width,
                          yerr=[yerr_lower, yerr_upper],
                          capsize=4,
                          label=label,
                          color=COLORS[region],
                          edgecolor='white',
                          linewidth=1,
                          error_kw={'linewidth': 1.8, 'capthick': 1.8, 'ecolor': '#333333'})

        ax.set_ylabel('$R^2$', fontsize=13, fontweight='medium')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12, fontweight='medium')
        ax.set_ylim(0, 0.55)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        ax.set_axisbelow(True)

        ax.legend(frameon=False, fontsize=11, loc='upper left')

        n_seeds = len(merged_results)
        ax.text(0.98, 0.02, f'n = {n_seeds} seeds, error bars = IQR', 
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                color='#666666')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'r2_comparison_mpi.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: r2_comparison_mpi.png")
        
        print("\n✓ All done!")


if __name__ == '__main__':
    main()
