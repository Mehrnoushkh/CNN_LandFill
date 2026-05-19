"""
Reproducible 3-method comparison with multiple seeds.thi sversion has better plots as an output
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
import gc

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

OUTPUT_DIR = './output_three_methods_multiseed_v03'

# Multiple seeds for statistical robustness
SEEDS = [8,65,788,2333,78] #[12,124,458,789,1120] #[42, 123, 456, 789, 1011]

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


def get_spatial_predictions(model, dataset, ocean_mask, device):
    """Get spatial map of predictions averaged over all time steps."""
    model.eval()
    S_mean = dataset.S_mean
    S_std = dataset.S_std
    all_preds = []
    all_targets = []
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out[:, 0:1, :, :].cpu().numpy() * S_std + S_mean
            target = y.cpu().numpy() * S_std + S_mean
            all_preds.append(pred)
            all_targets.append(target)
    preds = np.concatenate(all_preds, axis=0)[:, 0, :, :]
    targets = np.concatenate(all_targets, axis=0)[:, 0, :, :]
    pred_mean = np.mean(preds, axis=0)
    target_mean = np.mean(targets, axis=0)
    error_mean = pred_mean - target_mean
    pred_mean_masked = pred_mean.copy()
    target_mean_masked = target_mean.copy()
    error_mean_masked = error_mean.copy()
    pred_mean_masked[~ocean_mask] = np.nan
    target_mean_masked[~ocean_mask] = np.nan
    error_mean_masked[~ocean_mask] = np.nan
    return {
        'pred': pred_mean_masked,
        'target': target_mean_masked,
        'error': error_mean_masked,
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
    # Extract lat/lon (try common names)
    for lat_name in ['latitude', 'lat']:
        if lat_name in ds_orig:
            lat_hires = ds_orig[lat_name].values
            break
    for lon_name in ['longitude', 'lon']:
        if lon_name in ds_orig:
            lon_hires = ds_orig[lon_name].values
            break
    ds_orig.close()

    # Coarsen lat/lon to match coarsened data
    ny_trim = (len(lat_hires) // COARSEN_FACTOR) * COARSEN_FACTOR
    nx_trim = (len(lon_hires) // COARSEN_FACTOR) * COARSEN_FACTOR
    lat_coarse = lat_hires[:ny_trim].reshape(-1, COARSEN_FACTOR).mean(axis=1)
    lon_coarse = lon_hires[:nx_trim].reshape(-1, COARSEN_FACTOR).mean(axis=1)

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
        spatial_preds = {}

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

            # Spatial predictions (time-averaged over ALL data)
            spatial_preds[name] = get_spatial_predictions(model, dataset, ocean_mask, device)

            all_results[name]['overall'].append(results['r2_overall'])
            all_results[name]['coastal'].append(results['r2_coastal'])
            all_results[name]['open_ocean'].append(results['r2_open_ocean'])

            print(f"    R² = {results['r2_overall']:.4f}")

            del model
            gc.collect()

        # Save spatial predictions for this seed
        np.savez(os.path.join(OUTPUT_DIR, f'spatial_pred_seed_{seed}.npz'),
                 pred_zero=spatial_preds['zero_fill']['pred'],
                 pred_replicate=spatial_preds['replicate_fill']['pred'],
                 pred_laplace=spatial_preds['laplace_fill']['pred'],
                 target=spatial_preds['zero_fill']['target'])

    # =========================================================================
    # Compute statistics
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (mean ± std over {len(SEEDS)} seeds)")
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
    # Load spatial predictions and average across seeds
    # =========================================================================
    print("\nAveraging spatial predictions across seeds...")

    methods = ['zero_fill', 'replicate_fill', 'laplace_fill']
    method_labels = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']

    all_preds = {m: [] for m in methods}
    target_avg = None

    for seed in SEEDS:
        data = np.load(os.path.join(OUTPUT_DIR, f'spatial_pred_seed_{seed}.npz'))
        all_preds['zero_fill'].append(data['pred_zero'])
        all_preds['replicate_fill'].append(data['pred_replicate'])
        all_preds['laplace_fill'].append(data['pred_laplace'])
        if target_avg is None:
            target_avg = data['target']

    avg_preds = {m: np.nanmean(all_preds[m], axis=0) for m in methods}
    avg_errors = {m: avg_preds[m] - target_avg for m in methods}

    # Coordinate arrays for pcolormesh (need edges, not centers)
    lon_2d, lat_2d = np.meshgrid(lon_coarse, lat_coarse)

    # =========================================================================
    # Plot 1: Target vs Predictions (2x2) — seed-averaged
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    all_data = [target_avg] + [avg_preds[m] for m in methods]
    vmin, vmax = np.nanpercentile(np.concatenate([d.flatten() for d in all_data]), [2, 98])

    titles = ['True $S_T$ (Target)', 'Predicted $S_T$ (Zero-fill)',
              'Predicted $S_T$ (Replicate-fill)', 'Predicted $S_T$ (Laplace-fill)']
    data_list = [target_avg, avg_preds['zero_fill'],
                 avg_preds['replicate_fill'], avg_preds['laplace_fill']]

    for ax, plot_data, title in zip(axes.flat, data_list, titles):
        er_masked = plot_data.copy()
        er_masked[~ocean_mask] = np.nan 
        masked = np.ma.masked_invalid(er_masked)
        im = ax.pcolormesh(lon_2d, lat_2d, masked,
                           cmap='viridis', vmin=vmin, vmax=vmax, shading='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_xlim(lon_coarse.min(), lon_coarse.max())
        ax.set_ylim(lat_coarse.min(), lat_coarse.max())
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(True, alpha=0.3, linestyle='--')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label('Subgrid Variance $S_T$ (K²)', fontsize=11)
    fig.suptitle(f'Subgrid Temperature Variance (n={len(SEEDS)} seeds average)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ST_predictions.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'ST_predictions.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: ST_predictions.png/pdf")

    # =========================================================================
    # Plot 2: Errors (1x3) — seed-averaged
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    err_max = np.nanpercentile(np.abs(np.concatenate([avg_errors[m].flatten() for m in methods])), 98)

    for ax, method, label in zip(axes, methods, method_labels):
        err_masked = avg_errors[method].copy()
        err_masked[~ocean_mask] = np.nan       
        masked = np.ma.masked_invalid(err_masked)
        im = ax.pcolormesh(lon_2d, lat_2d, masked,
                           cmap='RdBu_r', vmin=-err_max, vmax=err_max, shading='nearest')
        ax.set_title(f'Error: {label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_xlim(lon_coarse.min(), lon_coarse.max())
        ax.set_ylim(lat_coarse.min(), lat_coarse.max())
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(True, alpha=0.3, linestyle='--')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('Error ($S_T^{pred} - S_T^{true}$) (K²)', fontsize=11)
    fig.suptitle(f'Prediction Errors (n={len(SEEDS)} seeds average)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ST_errors.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'ST_errors.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: ST_errors.png/pdf")

    # =========================================================================
    # Plot 3: Zoomed Coastal Region (Gulf of Mexico)
    # =========================================================================
    lat_min, lat_max = 15, 35
    lon_min, lon_max = -100, -75

    lat_idx = np.where((lat_coarse >= lat_min) & (lat_coarse <= lat_max))[0]
    lon_idx = np.where((lon_coarse >= lon_min) & (lon_coarse <= lon_max))[0]
    lat_zoom = lat_coarse[lat_idx]
    lon_zoom = lon_coarse[lon_idx]
    lon_z, lat_z = np.meshgrid(lon_zoom, lat_zoom)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Top row: Predictions
    zoom_preds = [avg_preds[m][np.ix_(lat_idx, lon_idx)] for m in methods]
    vmin_z, vmax_z = np.nanpercentile(np.concatenate([d.flatten() for d in zoom_preds]), [2, 98])
    ocean_mask_zoom = ocean_mask[np.ix_(lat_idx, lon_idx)]

    for ax, data, label in zip(axes[0], zoom_preds, method_labels):
        data[~ocean_mask_zoom] = np.nan 
        masked = np.ma.masked_invalid(data)
        im1 = ax.pcolormesh(lon_z, lat_z, masked,
                            cmap='viridis', vmin=vmin_z, vmax=vmax_z, shading='nearest')
        ax.set_title(f'Predicted $S_T$: {label}', fontsize=11)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3)

    # Bottom row: Errors
    zoom_errors = [avg_errors[m][np.ix_(lat_idx, lon_idx)] for m in methods]
    err_max_z = np.nanpercentile(np.abs(np.concatenate([d.flatten() for d in zoom_errors])), 98)

    for ax, data, label in zip(axes[1], zoom_errors, method_labels):
        masked = np.ma.masked_invalid(data)
        im2 = ax.pcolormesh(lon_z, lat_z, masked,
                            cmap='RdBu_r', vmin=-err_max_z, vmax=err_max_z, shading='nearest')
        ax.set_title(f'Error: {label}', fontsize=11)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3)

    fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8, label='$S_T$ (K²)')
    fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8, label='Error (K²)')
    fig.suptitle(f'Zoomed Coastal Region: Gulf of Mexico (n={len(SEEDS)} seeds avg)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ST_coastal_zoom.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'ST_coastal_zoom.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: ST_coastal_zoom.png/pdf")

    # =========================================================================
    # Plot 4: R² Bar Chart with error bars
    # =========================================================================
    COLORS = ['#3498DB', '#E67E22', '#2ECC71']

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(method_labels))
    width = 0.25
    regions = ['overall', 'coastal', 'open_ocean']
    region_labels = ['Overall', 'Coastal', 'Open Ocean']

    for i, (region, label, color) in enumerate(zip(regions, region_labels, COLORS)):
        means = [final_results[m][region]['mean'] for m in methods]
        stds = [final_results[m][region]['std'] for m in methods]

        bars = ax.bar(x + (i - 1) * width, means, width,
                      yerr=stds, capsize=4,
                      label=label,
                      color=color,
                      edgecolor='white',
                      linewidth=0.7,
                      error_kw={'linewidth': 1.2, 'capthick': 1.2})

        for j, (bar, mean_val) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.annotate(f'{mean_val:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height + stds[j] + 0.005),
                        ha='center', va='bottom', fontsize=8, fontweight='medium')

    ax.set_ylabel('$R^2$ Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=11)
    ax.set_ylim(0, max([final_results[m]['open_ocean']['mean'] for m in methods]) * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods_errorbars.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods_errorbars.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: r2_three_methods_errorbars.png/pdf")

    # =========================================================================
    # Plot 5: RMSE/MAE bar chart
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metric_data = {'RMSE': {}, 'MAE': {}}
    for method in methods:
        err = avg_errors[method]
        coastal_err = err[coastal_mask]
        open_err = err[ocean_mask & ~coastal_mask]

        metric_data['RMSE'][method] = {'coastal': np.sqrt(np.nanmean(coastal_err**2)),
                                       'open_ocean': np.sqrt(np.nanmean(open_err**2))}
        metric_data['MAE'][method] = {'coastal': np.nanmean(np.abs(coastal_err)),
                                      'open_ocean': np.nanmean(np.abs(open_err))}

    x = np.arange(3)
    width = 0.35
    bar_colors = {'coastal': '#D55E00', 'open_ocean': '#009E73'}

    for ax, metric_name in zip(axes, ['RMSE', 'MAE']):
        coastal_vals = [metric_data[metric_name][m]['coastal'] for m in methods]
        ocean_vals = [metric_data[metric_name][m]['open_ocean'] for m in methods]

        ax.bar(x - width/2, coastal_vals, width, label='Coastal', color=bar_colors['coastal'])
        ax.bar(x + width/2, ocean_vals, width, label='Open Ocean', color=bar_colors['open_ocean'])

        ax.set_ylabel(f'{metric_name} (K²)')
        ax.set_title(f'{metric_name} by Region')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Spatial Error Metrics (n={len(SEEDS)} seeds avg)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ST_error_metrics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'ST_error_metrics.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: ST_error_metrics.png/pdf")

    # Save target and masks for future use
    np.savez(os.path.join(OUTPUT_DIR, 'target_and_masks.npz'),
             target=target_avg, ocean_mask=ocean_mask, coastal_mask=coastal_mask,
             lat=lat_coarse, lon=lon_coarse)

    print(f"\n{'='*60}")
    print(f"✓ All outputs saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")
    print("Done!")


if __name__ == '__main__':
    main()
