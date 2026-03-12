"""
FAIR COMPARISON: Three land fill methods for CNN training
- Zero-fill: land = 0
- Replicate-fill: land = nearest ocean value
- Laplace-fill: physics-based harmonic extension

All three predict the SAME ocean-only S_T target.
Memory-efficient version using float32 and chunked processing.
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

torch.set_num_threads(32)

# =============================================================================
# SETTINGS
# =============================================================================

GZ21_PATH = '.'

# Data files
ORIGINAL_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc'
FILLED_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/Neumann_DrichletBc/cplusplus/fordays/sst_filled_365days_2023.nc'

# Variable names
ORIGINAL_VAR_NAME = 'analysed_sst'
FILLED_VAR_NAME = 'sst_neumann'
FILL_VALUE = -999.0

# Settings
COARSEN_FACTOR = 4
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
TRAIN_FRACTION = 0.8
MAX_TIME_STEPS = 364

OUTPUT_DIR = './output_three_methods'

# =============================================================================
# Import GZ21 model
# =============================================================================

sys.path.insert(0, GZ21_PATH)

try:
    from models.models1 import FullyCNN
    print("✓ Imported GZ21 FullyCNN")
except ImportError as e:
    print(f"✗ Could not import GZ21: {e}")
    sys.exit(1)

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
# Functions (Memory Efficient)
# =============================================================================

def compute_subgrid_forcing(T_highres, factor, chunk_size=50):
    """
    S_T = coarsen(T²) - coarsen(T)²
    Memory-efficient version: process in time chunks using float32.
    """
    n_times, ny, nx = T_highres.shape
    ny_trim = (ny // factor) * factor
    nx_trim = (nx // factor) * factor
    
    ny_coarse = ny_trim // factor
    nx_coarse = nx_trim // factor
    
    # Allocate output arrays (float32)
    T_coarse = np.zeros((n_times, ny_coarse, nx_coarse), dtype=np.float32)
    S_T = np.zeros((n_times, ny_coarse, nx_coarse), dtype=np.float32)
    
    print(f"    Coarsening {n_times} time steps in chunks of {chunk_size}...")
    
    # Process in chunks
    for t_start in range(0, n_times, chunk_size):
        t_end = min(t_start + chunk_size, n_times)
        
        # Get chunk (float32)
        T_chunk = T_highres[t_start:t_end, :ny_trim, :nx_trim].astype(np.float32)
        
        # Reshape for coarsening
        nt_chunk = t_end - t_start
        T_reshaped = T_chunk.reshape(nt_chunk, ny_coarse, factor, nx_coarse, factor)
        
        # Compute coarse T and T²
        T_coarse[t_start:t_end] = np.nanmean(T_reshaped, axis=(2, 4))
        T_sq_coarse = np.nanmean(T_reshaped ** 2, axis=(2, 4))
        
        # Subgrid variance
        S_T[t_start:t_end] = T_sq_coarse - T_coarse[t_start:t_end] ** 2
        
        # Free memory
        del T_chunk, T_reshaped, T_sq_coarse
        
        if t_start % 100 == 0:
            print(f"      Processed {t_start}/{n_times} time steps")
    
    print(f"    Done. Output shape: {T_coarse.shape}")
    return T_coarse, S_T


def create_replicate_fill(T_with_nan, land_mask_3d):
    """
    Replicate padding: fill land with nearest ocean value.
    Memory-efficient version using float32.
    """
    print("  Creating replicate-fill...")
    n_times, ny, nx = T_with_nan.shape
    
    # Use float32 to save memory
    T_replicate = T_with_nan.astype(np.float32).copy()
    
    # Get 2D land mask (land doesn't change in time)
    land_mask_2d = land_mask_3d[0]
    
    # Get indices of nearest ocean point
    print("    Computing nearest ocean indices...")
    _, indices = distance_transform_edt(land_mask_2d, return_indices=True)
    
    # Pre-compute the indices for land points
    land_rows = indices[0][land_mask_2d]
    land_cols = indices[1][land_mask_2d]
    
    # Apply to all time steps
    print(f"    Filling {n_times} time steps...")
    for t in range(n_times):
        T_replicate[t, land_mask_2d] = T_with_nan[t, land_rows, land_cols]
        
        if t % 100 == 0:
            print(f"      Time step {t}/{n_times}")
    
    # Handle remaining NaN
    T_replicate = np.nan_to_num(T_replicate, nan=288.0)
    
    print(f"    Done. Range: [{T_replicate.min():.1f}, {T_replicate.max():.1f}] K")
    return T_replicate


def train_model(train_loader, val_loader, mask_tensor, device, method_name):
    """Train a model and return results."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {method_name}")
    print(f"{'='*60}")

    model = FullyCNN(n_in_channels=1, n_out_channels=2, padding='same')
    model.final_transformation = SoftPlusTransform(n_targets=1)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = HeteroskedasticGaussianLoss()

    best_loss = float('inf')
    best_state = None
    history = {'train': [], 'val': []}

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    for epoch in range(EPOCHS):
        # Train
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

        # Validate
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
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, history, best_loss


def evaluate_model(model, data_loader, ocean_mask, coastal_mask, S_mean, S_std, device):
    """Evaluate model and return predictions, truths, and metrics."""
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

    # Overall ocean
    r2_overall = compute_r2(ocean_mask)
    
    # Coastal
    r2_coastal = compute_r2(coastal_mask)
    
    # Open ocean
    open_ocean_mask = ocean_mask & ~coastal_mask
    r2_open = compute_r2(open_ocean_mask)

    return {
        'r2_overall': r2_overall,
        'r2_coastal': r2_coastal,
        'r2_open_ocean': r2_open,
        'predictions': preds,
        'targets': targets
    }


def main():
    print("=" * 60)
    print("THREE-WAY COMPARISON (Memory Efficient)")
    print("Zero-fill vs Replicate-fill vs Laplace-fill")
    print("All predict the SAME ocean-only S_T target")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Load ORIGINAL data (float32 to save memory)
    # =========================================================================
    print(f"\n{'='*60}")
    print("LOADING ORIGINAL DATA")
    print(f"{'='*60}")

    ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
    T_orig = ds_orig[ORIGINAL_VAR_NAME].values[:MAX_TIME_STEPS, :, :].astype(np.float32)
    ds_orig.close()
    
    print(f"Shape: {T_orig.shape}")
    print(f"Memory: {T_orig.nbytes / 1e9:.2f} GB")

    # Create masks
    land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
    print(f"Land fraction: {land_mask.mean():.1%}")

    # T with NaN over land
    T_orig_with_nan = T_orig.copy()
    T_orig_with_nan[land_mask] = np.nan

    # =========================================================================
    # Create THREE versions
    # =========================================================================
    print(f"\n{'='*60}")
    print("CREATING THREE FILLED VERSIONS")
    print(f"{'='*60}")

    # 1. Zero-fill
    print("\n1. Zero-fill (land = 0)...")
    T_zero = np.nan_to_num(T_orig_with_nan, nan=0.0).astype(np.float32)
    print(f"   Range: [{T_zero.min():.1f}, {T_zero.max():.1f}] K")

    # 2. Replicate-fill
    print("\n2. Replicate-fill (land = nearest ocean)...")
    T_replicate = create_replicate_fill(T_orig_with_nan, land_mask)

    # 3. Laplace-fill
    print("\n3. Laplace-fill (from file)...")
    ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
    T_laplace = ds_filled[FILLED_VAR_NAME].values[:MAX_TIME_STEPS, 0, :, :].astype(np.float32)
    ds_filled.close()
    print(f"   Range: [{T_laplace.min():.1f}, {T_laplace.max():.1f}] K")

    # =========================================================================
    # Compute coarse data (one at a time to save memory)
    # =========================================================================
    print(f"\n{'='*60}")
    print("COMPUTING COARSE DATA AND TARGET")
    print(f"{'='*60}")

    # TRUE target first (ocean-only)
    print("\nComputing TRUE target (from original with NaN)...")
    T_coarse_orig, S_T_target = compute_subgrid_forcing(T_orig_with_nan, COARSEN_FACTOR)
    ocean_mask = np.isfinite(T_coarse_orig[0])
    print(f"Ocean points: {ocean_mask.sum()}")

    # Coastal mask (within 10 cells of land)
    land_mask_coarse = ~ocean_mask
    coastal_mask = binary_dilation(land_mask_coarse, iterations=10) & ocean_mask
    print(f"Coastal points: {coastal_mask.sum()}")
    print(f"Open ocean points: {(ocean_mask & ~coastal_mask).sum()}")

    # Target for all methods
    S_T_target_masked = S_T_target.copy()
    S_T_target_masked[:, ~ocean_mask] = 0.0

    # Free memory
    del T_coarse_orig, T_orig_with_nan
    
    # Coarse zero-fill
    print("\nComputing coarse zero-fill...")
    T_coarse_zero, _ = compute_subgrid_forcing(T_zero, COARSEN_FACTOR)
    del T_zero  # Free memory
    
    # Coarse replicate
    print("\nComputing coarse replicate-fill...")
    T_coarse_replicate, _ = compute_subgrid_forcing(T_replicate, COARSEN_FACTOR)
    del T_replicate  # Free memory
    
    # Coarse laplace
    print("\nComputing coarse Laplace-fill...")
    T_coarse_laplace, _ = compute_subgrid_forcing(T_laplace, COARSEN_FACTOR)
    del T_laplace  # Free memory

    print(f"\nCoarse shape: {T_coarse_zero.shape}")

    # =========================================================================
    # Create datasets
    # =========================================================================
    print(f"\n{'='*60}")
    print("CREATING DATASETS")
    print(f"{'='*60}")

    dataset_zero = TemperatureDataset(T_coarse_zero, S_T_target_masked, ocean_mask)
    dataset_replicate = TemperatureDataset(T_coarse_replicate, S_T_target_masked, ocean_mask)
    dataset_laplace = TemperatureDataset(T_coarse_laplace, S_T_target_masked, ocean_mask)

    # Free memory
    del T_coarse_zero, T_coarse_replicate, T_coarse_laplace, S_T_target, S_T_target_masked

    # Same split for all (fair comparison)
    n = len(dataset_zero)
    train_size = int(TRAIN_FRACTION * n)
    val_size = n - train_size

    generator = torch.Generator().manual_seed(42)
    train_zero, val_zero = random_split(dataset_zero, [train_size, val_size], generator=generator)
    
    generator = torch.Generator().manual_seed(42)
    train_replicate, val_replicate = random_split(dataset_replicate, [train_size, val_size], generator=generator)
    
    generator = torch.Generator().manual_seed(42)
    train_laplace, val_laplace = random_split(dataset_laplace, [train_size, val_size], generator=generator)

    # Data loaders
    train_loader_zero = DataLoader(train_zero, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_zero = DataLoader(val_zero, batch_size=BATCH_SIZE, num_workers=0)

    train_loader_replicate = DataLoader(train_replicate, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_replicate = DataLoader(val_replicate, batch_size=BATCH_SIZE, num_workers=0)

    train_loader_laplace = DataLoader(train_laplace, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_laplace = DataLoader(val_laplace, batch_size=BATCH_SIZE, num_workers=0)

    print(f"Train: {train_size}, Val: {val_size}")

    # Mask tensor
    mask_tensor = torch.tensor(ocean_mask, dtype=torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # =========================================================================
    # Train THREE models
    # =========================================================================

    model_zero, history_zero, _ = train_model(
        train_loader_zero, val_loader_zero, mask_tensor, device, "ZERO-FILL"
    )

    model_replicate, history_replicate, _ = train_model(
        train_loader_replicate, val_loader_replicate, mask_tensor, device, "REPLICATE-FILL"
    )

    model_laplace, history_laplace, _ = train_model(
        train_loader_laplace, val_loader_laplace, mask_tensor, device, "LAPLACE-FILL"
    )

    # =========================================================================
    # Evaluate THREE models
    # =========================================================================
    print(f"\n{'='*60}")
    print("EVALUATING MODELS")
    print(f"{'='*60}")

    results_zero = evaluate_model(
        model_zero, val_loader_zero, ocean_mask, coastal_mask,
        dataset_zero.S_mean, dataset_zero.S_std, device
    )

    results_replicate = evaluate_model(
        model_replicate, val_loader_replicate, ocean_mask, coastal_mask,
        dataset_replicate.S_mean, dataset_replicate.S_std, device
    )

    results_laplace = evaluate_model(
        model_laplace, val_loader_laplace, ocean_mask, coastal_mask,
        dataset_laplace.S_mean, dataset_laplace.S_std, device
    )

    # =========================================================================
    # Print comparison
    # =========================================================================
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{'Method':<20} {'Overall R²':>12} {'Coastal R²':>12} {'Open Ocean R²':>14}")
    print("-" * 62)
    print(f"{'Zero-fill':<20} {results_zero['r2_overall']:>12.4f} {results_zero['r2_coastal']:>12.4f} {results_zero['r2_open_ocean']:>14.4f}")
    print(f"{'Replicate-fill':<20} {results_replicate['r2_overall']:>12.4f} {results_replicate['r2_coastal']:>12.4f} {results_replicate['r2_open_ocean']:>14.4f}")
    print(f"{'Laplace-fill':<20} {results_laplace['r2_overall']:>12.4f} {results_laplace['r2_coastal']:>12.4f} {results_laplace['r2_open_ocean']:>14.4f}")

    # =========================================================================
    # Save models
    # =========================================================================
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print(f"{'='*60}")

    for name, model, dataset in [
        ('zero', model_zero, dataset_zero),
        ('replicate', model_replicate, dataset_replicate),
        ('laplace', model_laplace, dataset_laplace)
    ]:
        torch.save({
            'model_state_dict': model.state_dict(),
            'T_mean': dataset.T_mean,
            'T_std': dataset.T_std,
            'S_mean': dataset.S_mean,
            'S_std': dataset.S_std,
        }, os.path.join(OUTPUT_DIR, f'model_{name}_fill.pth'))
        print(f"  ✓ Saved model_{name}_fill.pth")

    # =========================================================================
    # Generate plots
    # =========================================================================
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.unicode_minus': False  # Ensures minus signs in R^2 scores render correctly
    })

# Your figure size for side-by-side LaTeX columns
    fig, ax = plt.subplots(figsize=(6, 5))
    # Plot 1: R² comparison by region
    #fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
    x = np.arange(len(methods))
    width = 0.25

    overall = [results_zero['r2_overall'], results_replicate['r2_overall'], results_laplace['r2_overall']]
    coastal = [results_zero['r2_coastal'], results_replicate['r2_coastal'], results_laplace['r2_coastal']]
    open_ocean = [results_zero['r2_open_ocean'], results_replicate['r2_open_ocean'], results_laplace['r2_open_ocean']]

    #bars1 = ax.bar(x - width, overall, width, label='Overall', color='steelblue')
    #bars2 = ax.bar(x, coastal, width, label='Coastal', color='coral')
    #bars3 = ax.bar(x + width, open_ocean, width, label='Open Ocean', color='seagreen')
    bars1 = ax.bar(x - width, overall, width, label='Overall', color='#2E5A88')
    bars2 = ax.bar(x, coastal, width, label='Coastal', color='#D97B42')
    bars3 = ax.bar(x + width, open_ocean, width, label='Open Ocean', color='#4F9D69')
    ax.set_ylabel('R² Score')
    #ax.set_title('CNN Performance: Three Land Fill Methods\n(All predict same ocean-only target)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(overall), max(coastal), max(open_ocean)) * 1.2)

    # Add values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods.png'), dpi=150)
    plt.close()
    print("  ✓ Saved r2_three_methods.png")

    # Plot 2: Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, history, name in [
        (axes[0], history_zero, 'Zero-fill'),
        (axes[1], history_replicate, 'Replicate-fill'),
        (axes[2], history_laplace, 'Laplace-fill')
    ]:
        ax.plot(history['train'], label='Train', color='blue')
        ax.plot(history['val'], label='Val', color='orange')
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Loss Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_three_methods.png'), dpi=150)
    plt.close()
    print("  ✓ Saved training_three_methods.png")

    # Plot 3: Map showing regions
    fig, ax = plt.subplots(figsize=(12, 5))
    
    region_map = np.zeros_like(ocean_mask, dtype=float)
    region_map[~ocean_mask] = np.nan  # Land = NaN (white)
    region_map[ocean_mask & ~coastal_mask] = 0  # Open ocean = 0 (blue)
    region_map[coastal_mask] = 1  # Coastal = 1 (red)
    
    im = ax.imshow(region_map, origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    ax.set_title('Coastal (red) vs Open Ocean (blue) Regions')
    ax.set_xlabel('Longitude index')
    ax.set_ylabel('Latitude index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'region_map.png'), dpi=150)
    plt.close()
    print("  ✓ Saved region_map.png")

    # Save results to JSON
    results_json = {
        'zero_fill': {
            'r2_overall': float(results_zero['r2_overall']),
            'r2_coastal': float(results_zero['r2_coastal']),
            'r2_open_ocean': float(results_zero['r2_open_ocean']),
        },
        'replicate_fill': {
            'r2_overall': float(results_replicate['r2_overall']),
            'r2_coastal': float(results_replicate['r2_coastal']),
            'r2_open_ocean': float(results_replicate['r2_open_ocean']),
        },
        'laplace_fill': {
            'r2_overall': float(results_laplace['r2_overall']),
            'r2_coastal': float(results_laplace['r2_coastal']),
            'r2_open_ocean': float(results_laplace['r2_open_ocean']),
        },
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    print("  ✓ Saved results.json")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nOverall R²:")
    print(f"  Zero-fill:      {results_zero['r2_overall']:.4f}")
    print(f"  Replicate-fill: {results_replicate['r2_overall']:.4f}")
    print(f"  Laplace-fill:   {results_laplace['r2_overall']:.4f}")
    
    print(f"\nImprovement over Zero-fill:")
    print(f"  Replicate: {(results_replicate['r2_overall']/results_zero['r2_overall'] - 1)*100:+.1f}% overall, {(results_replicate['r2_coastal']/results_zero['r2_coastal'] - 1)*100:+.1f}% coastal")
    print(f"  Laplace:   {(results_laplace['r2_overall']/results_zero['r2_overall'] - 1)*100:+.1f}% overall, {(results_laplace['r2_coastal']/results_zero['r2_coastal'] - 1)*100:+.1f}% coastal")
    
    print(f"\nImprovement of Laplace over Replicate:")
    print(f"  Overall: {(results_laplace['r2_overall']/results_replicate['r2_overall'] - 1)*100:+.1f}%")
    print(f"  Coastal: {(results_laplace['r2_coastal']/results_replicate['r2_coastal'] - 1)*100:+.1f}%")
    print(f"  Open Ocean: {(results_laplace['r2_open_ocean']/results_replicate['r2_open_ocean'] - 1)*100:+.1f}%")

    print("\nDone!")


if __name__ == '__main__':
    main()
