"""
compare_maps.py

Generate side-by-side comparison maps:
- Zero-fill vs Physics-fill predictions
- Show where physics-fill does better

Usage:
    python compare_maps.py
"""

import sys
import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# =============================================================================
# SETTINGS
# =============================================================================

GZ21_PATH = '.'

# Data files
#ORIGINAL_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc'
#FILLED_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/Neumann_DrichletBc/cplusplus/sst_filled_100_2023.nc'
ORIGINAL_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc'
FILLED_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/Neumann_DrichletBc/cplusplus/fordays/sst_filled_365days_2023.nc'


# Model files (from compare_methods.py output)
MODEL_ZERO_FILE = 'output_comparison/model_zero_fill.pth'
MODEL_PHYSICS_FILE = 'output_comparison/model_physics_fill.pth'

OUTPUT_DIR = 'output_comparison_1year'

# Variable names
ORIGINAL_VAR_NAME = 'analysed_sst'
FILLED_VAR_NAME = 'sst_neumann'
FILL_VALUE = -999.0

COARSEN_FACTOR = 4
MAX_TIME_STEPS = 364

# =============================================================================
# Import GZ21 model
# =============================================================================

sys.path.insert(0, GZ21_PATH)
from models.models1 import FullyCNN

class SoftPlusTransform(nn.Module):
    def __init__(self, n_targets=1):
        super().__init__()
        self.n_targets = n_targets

    def forward(self, x):
        mean = x[:, :self.n_targets, :, :]
        log_std = x[:, self.n_targets:, :, :]
        std = F.softplus(log_std) + 1e-6
        return torch.cat([mean, std], dim=1)

# =============================================================================
# Functions
# =============================================================================

def compute_subgrid_forcing(T_highres, factor):
    """S_T = coarsen(T²) - coarsen(T)²"""
    n_times, ny, nx = T_highres.shape
    ny_trim = (ny // factor) * factor
    nx_trim = (nx // factor) * factor
    T = T_highres[:, :ny_trim, :nx_trim]

    T_reshaped = T.reshape(n_times, ny_trim // factor, factor, nx_trim // factor, factor)
    T_coarse = np.nanmean(T_reshaped, axis=(2, 4))
    T_sq_coarse = np.nanmean(T_reshaped ** 2, axis=(2, 4))
    S_T = T_sq_coarse - T_coarse ** 2

    return T_coarse, S_T


def load_model(model_path, device):
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = FullyCNN(n_in_channels=1, n_out_channels=2, padding='same')
    model.final_transformation = SoftPlusTransform(n_targets=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def predict(model, T_coarse, T_mean, T_std, S_mean, S_std, device):
    """Make predictions for all time steps."""
    predictions = []
    
    with torch.no_grad():
        for t in range(T_coarse.shape[0]):
            # Normalize input
            T_input = (T_coarse[t] - T_mean) / T_std
            T_input = torch.tensor(T_input, dtype=torch.float32)
            T_input = T_input.unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict
            output = model(T_input)
            
            # Denormalize prediction
            pred = output[0, 0].cpu().numpy() * S_std + S_mean
            predictions.append(pred)
    
    return np.array(predictions)


def main():
    print("=" * 60)
    print("COMPARE PREDICTION MAPS")
    print("Zero-fill vs Physics-fill")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # Load models
    # =========================================================================
    print("\nLoading models...")
    
    model_zero, ckpt_zero = load_model(MODEL_ZERO_FILE, device)
    print(f"  ✓ Loaded zero-fill model")
    
    model_physics, ckpt_physics = load_model(MODEL_PHYSICS_FILE, device)
    print(f"  ✓ Loaded physics-fill model")

    # =========================================================================
    # Load data
    # =========================================================================
    print("\nLoading data...")
    
    # Original data
    ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
    T_orig = ds_orig[ORIGINAL_VAR_NAME].values[:MAX_TIME_STEPS, :, :]
    
    # Create masks
    land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
    T_orig_with_nan = T_orig.copy().astype(float)
    T_orig_with_nan[land_mask] = np.nan
    T_zero_filled = np.nan_to_num(T_orig_with_nan, nan=0.0)
    
    # Filled data
    ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
    T_physics_filled = ds_filled[FILLED_VAR_NAME].values[:MAX_TIME_STEPS, 0, :, :]
    
    print(f"  Original shape: {T_orig.shape}")
    print(f"  Filled shape: {T_physics_filled.shape}")

    # =========================================================================
    # Compute coarse data
    # =========================================================================
    print("\nComputing coarse data...")
    
    T_coarse_zero, _ = compute_subgrid_forcing(T_zero_filled, COARSEN_FACTOR)
    T_coarse_physics, _ = compute_subgrid_forcing(T_physics_filled, COARSEN_FACTOR)
    T_coarse_orig, S_T_true = compute_subgrid_forcing(T_orig_with_nan, COARSEN_FACTOR)
    
    ocean_mask = np.isfinite(T_coarse_orig[0])
    
    # Create coastal mask (within 10 grid cells of land)
    land_mask_coarse = ~ocean_mask
    coastal_mask = binary_dilation(land_mask_coarse, iterations=10) & ocean_mask
    open_ocean_mask = ocean_mask & ~coastal_mask
    
    print(f"  Coarse shape: {T_coarse_zero.shape}")
    print(f"  Ocean points: {ocean_mask.sum()}")
    print(f"  Coastal points: {coastal_mask.sum()}")
    print(f"  Open ocean points: {open_ocean_mask.sum()}")

    # =========================================================================
    # Make predictions
    # =========================================================================
    print("\nMaking predictions...")
    
    pred_zero = predict(model_zero, T_coarse_zero,
                        ckpt_zero['T_mean'], ckpt_zero['T_std'],
                        ckpt_zero['S_mean'], ckpt_zero['S_std'], device)
    print(f"  ✓ Zero-fill predictions done")
    
    pred_physics = predict(model_physics, T_coarse_physics,
                           ckpt_physics['T_mean'], ckpt_physics['T_std'],
                           ckpt_physics['S_mean'], ckpt_physics['S_std'], device)
    print(f"  ✓ Physics-fill predictions done")

    # =========================================================================
    # Compute errors
    # =========================================================================
    error_zero = pred_zero - S_T_true
    error_physics = pred_physics - S_T_true
    
    # Absolute errors
    abs_error_zero = np.abs(error_zero)
    abs_error_physics = np.abs(error_physics)
    
    # Where physics-fill is better
    physics_better = abs_error_physics < abs_error_zero

    # =========================================================================
    # Plot 1: Sample day comparison
    # =========================================================================
    print("\nGenerating plots...")
    
    sample_days = [0, 25, 50, 75]
    
    for day in sample_days:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        truth = S_T_true[day].copy()
        pred_z = pred_zero[day].copy()
        pred_p = pred_physics[day].copy()
        err_z = error_zero[day].copy()
        err_p = error_physics[day].copy()
        
        # Mask land
        truth[~ocean_mask] = np.nan
        pred_z[~ocean_mask] = np.nan
        pred_p[~ocean_mask] = np.nan
        err_z[~ocean_mask] = np.nan
        err_p[~ocean_mask] = np.nan
        
        # Common scales
        vmin = np.nanpercentile(truth, 2)
        vmax = np.nanpercentile(truth, 98)
        err_max = np.nanpercentile(np.abs(np.concatenate([err_z[ocean_mask], err_p[ocean_mask]])), 95)
        
        # Row 1: Zero-fill
        im = axes[0, 0].imshow(truth, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f'Day {day}: Truth (S_T)', fontsize=12)
        axes[0, 0].axis('off')
        plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
        
        im = axes[0, 1].imshow(pred_z, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f'Zero-fill Prediction', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        im = axes[0, 2].imshow(err_z, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
        axes[0, 2].set_title(f'Zero-fill Error', fontsize=12)
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        im = axes[0, 3].imshow(np.abs(err_z), cmap='Reds', vmin=0, vmax=err_max)
        axes[0, 3].set_title(f'Zero-fill |Error|', fontsize=12)
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
        
        # Row 2: Physics-fill
        im = axes[1, 0].imshow(truth, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f'Day {day}: Truth (S_T)', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        im = axes[1, 1].imshow(pred_p, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f'Physics-fill Prediction', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        
        im = axes[1, 2].imshow(err_p, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
        axes[1, 2].set_title(f'Physics-fill Error', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        
        im = axes[1, 3].imshow(np.abs(err_p), cmap='Reds', vmin=0, vmax=err_max)
        axes[1, 3].set_title(f'Physics-fill |Error|', fontsize=12)
        axes[1, 3].axis('off')
        plt.colorbar(im, ax=axes[1, 3], fraction=0.046)
        
        plt.suptitle(f'Day {day} Comparison: Zero-fill (top) vs Physics-fill (bottom)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'map_comparison_day{day}.png'), dpi=150)
        plt.close()
        print(f"  ✓ Saved map_comparison_day{day}.png")

    # =========================================================================
    # Plot 2: Average error maps
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    mean_abs_err_zero = np.nanmean(abs_error_zero, axis=0)
    mean_abs_err_physics = np.nanmean(abs_error_physics, axis=0)
    improvement = mean_abs_err_zero - mean_abs_err_physics  # Positive = physics better
    
    mean_abs_err_zero[~ocean_mask] = np.nan
    mean_abs_err_physics[~ocean_mask] = np.nan
    improvement[~ocean_mask] = np.nan
    
    vmax_err = np.nanpercentile(np.concatenate([mean_abs_err_zero[ocean_mask], 
                                                 mean_abs_err_physics[ocean_mask]]), 95)
    
    im = axes[0].imshow(mean_abs_err_zero, cmap='Reds', vmin=0, vmax=vmax_err)
    axes[0].set_title('Zero-fill: Mean |Error|', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    
    im = axes[1].imshow(mean_abs_err_physics, cmap='Reds', vmin=0, vmax=vmax_err)
    axes[1].set_title('Physics-fill: Mean |Error|', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    imp_max = np.nanpercentile(np.abs(improvement[ocean_mask]), 95)
    im = axes[2].imshow(improvement, cmap='RdBu_r', vmin=-imp_max, vmax=imp_max)
    axes[2].set_title('Improvement (Blue = Physics Better)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.suptitle('Average Error Comparison (All Days)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mean_error_comparison.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved mean_error_comparison.png")

    # =========================================================================
    # Plot 3: Coastal vs Open Ocean
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute regional R²
    def compute_r2(pred, truth, mask):
        p = pred[:, mask].flatten()
        t = truth[:, mask].flatten()
        valid = np.isfinite(p) & np.isfinite(t)
        p, t = p[valid], t[valid]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)
    
    r2_zero_coastal = compute_r2(pred_zero, S_T_true, coastal_mask)
    r2_zero_open = compute_r2(pred_zero, S_T_true, open_ocean_mask)
    r2_physics_coastal = compute_r2(pred_physics, S_T_true, coastal_mask)
    r2_physics_open = compute_r2(pred_physics, S_T_true, open_ocean_mask)
    
    # Bar chart
    x = np.arange(2)
    width = 0.35
    
    zero_r2 = [r2_zero_coastal, r2_zero_open]
    physics_r2 = [r2_physics_coastal, r2_physics_open]
    
    bars1 = axes[0].bar(x - width/2, zero_r2, width, label='Zero-fill', color='steelblue')
    bars2 = axes[0].bar(x + width/2, physics_r2, width, label='Laplace fill', color='coral')
    
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² by Region')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Coastal\n(within 10 cells of land)', 'Open Ocean'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
    
    # Show coastal mask
    mask_display = np.zeros_like(ocean_mask, dtype=float)
    mask_display[~ocean_mask] = np.nan  # Land
    mask_display[open_ocean_mask] = 0   # Open ocean
    mask_display[coastal_mask] = 1      # Coastal
    
    im = axes[1].imshow(mask_display, cmap='coolwarm', vmin=0, vmax=1)
    axes[1].set_title('Coastal (red) vs Open Ocean (blue)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'coastal_vs_open_ocean.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved coastal_vs_open_ocean.png")

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("REGIONAL R² COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Region':<25} {'Zero-fill':>12} {'Physics-fill':>12} {'Winner':>12}")
    print("-" * 65)
    print(f"{'Coastal (near land)':<25} {r2_zero_coastal:>12.4f} {r2_physics_coastal:>12.4f} {'Physics ✓' if r2_physics_coastal > r2_zero_coastal else 'Zero'}")
    print(f"{'Open Ocean':<25} {r2_zero_open:>12.4f} {r2_physics_open:>12.4f} {'Physics ✓' if r2_physics_open > r2_zero_open else 'Zero'}")
    
    print(f"\n{'='*60}")
    print("OUTPUTS SAVED")
    print(f"{'='*60}")
    print(f"  - map_comparison_day*.png  (4 files)")
    print(f"  - mean_error_comparison.png")
    print(f"  - coastal_vs_open_ocean.png")


if __name__ == '__main__':
    main()
