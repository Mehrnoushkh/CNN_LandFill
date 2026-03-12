"""
visualize_filled.py

Load your trained PHYSICS-FILLED model and compare predictions vs truth.

Usage:
    python visualize_filled.py
"""

import sys
import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================

GZ21_PATH = '.'

# Physics-filled data file
FILLED_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/Neumann_DrichletBc/cplusplus/sst_filled_30day_2023.nc'

# Original data file (for ocean mask and true S_T)
ORIGINAL_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc'

# Model file
MODEL_FILE = 'output_filled/model_sst_neumann.pth'
OUTPUT_DIR = 'output_filled'

# Variable names
FILLED_VAR_NAME = 'sst_neumann'
ORIGINAL_VAR_NAME = 'analysed_sst'
FILL_VALUE = -999.0

COARSEN_FACTOR = 4
MAX_TIME_STEPS = 30

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


def main():
    print("=" * 60)
    print("VISUALIZE PREDICTIONS (Physics-Filled Model)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Load trained model
    # -------------------------------------------------------------------------
    print(f"\nLoading model from {MODEL_FILE}...")
    checkpoint = torch.load(MODEL_FILE, map_location=device)

    model = FullyCNN(n_in_channels=1, n_out_channels=2, padding='same')
    model.final_transformation = SoftPlusTransform(n_targets=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get normalization values
    T_mean = checkpoint['T_mean']
    T_std = checkpoint['T_std']
    S_mean = checkpoint['S_mean']
    S_std = checkpoint['S_std']

    print(f"  T normalization: mean={T_mean:.2f}, std={T_std:.2f}")
    print(f"  S normalization: mean={S_mean:.4f}, std={S_std:.4f}")
    print("  ✓ Model loaded!")

    # -------------------------------------------------------------------------
    # Load FILLED data (input to model)
    # -------------------------------------------------------------------------
    print(f"\nLoading FILLED data from {FILLED_DATA_FILE}...")
    ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
    
    # Get filled temperature - squeeze depth dimension!
    T_filled = ds_filled[FILLED_VAR_NAME].values  # (time, depth, lat, lon)
    T_filled = T_filled[:, 0, :, :]               # (time, lat, lon)
    T_filled = T_filled[:MAX_TIME_STEPS, :, :]
    
    print(f"  Filled shape: {T_filled.shape}")

    # -------------------------------------------------------------------------
    # Load ORIGINAL data (ONLY for ocean mask)
    # -------------------------------------------------------------------------
    print(f"\nLoading ORIGINAL data from {ORIGINAL_DATA_FILE}...")
    print("  (Only using this for ocean mask)")
    ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
    T_orig = ds_orig[ORIGINAL_VAR_NAME].values
    T_orig = T_orig[:MAX_TIME_STEPS, :, :]

    # Create ocean mask from original (where data is valid, not -999)
    land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
    T_orig_with_nan = T_orig.copy().astype(float)
    T_orig_with_nan[land_mask] = np.nan

    # -------------------------------------------------------------------------
    # Compute coarse data and forcing (ALL from FILLED file)
    # -------------------------------------------------------------------------
    print("\nComputing coarse data and forcing...")
    
    # Coarse FILLED (input AND target come from filled file)
    T_coarse_filled, S_T_true = compute_subgrid_forcing(T_filled, COARSEN_FACTOR)
    
    # Get ocean mask from original (just to know where ocean is)
    T_coarse_orig, _ = compute_subgrid_forcing(T_orig_with_nan, COARSEN_FACTOR)
    ocean_mask = np.isfinite(T_coarse_orig[0])

    print(f"  Coarse shape: {T_coarse_filled.shape}")
    print(f"  Ocean points: {ocean_mask.sum()}")

    # -------------------------------------------------------------------------
    # Make predictions
    # -------------------------------------------------------------------------
    print("\nMaking predictions...")

    # Sample days (adjusted for 30 days max)
    sample_days = [0, 10, 20, 29]
    sample_days = [d for d in sample_days if d < T_coarse_filled.shape[0]]

    predictions = []
    uncertainties = []

    with torch.no_grad():
        for day in sample_days:
            # Prepare input (use FILLED data)
            T_input = (T_coarse_filled[day] - T_mean) / T_std
            T_input = torch.tensor(T_input, dtype=torch.float32)
            T_input = T_input.unsqueeze(0).unsqueeze(0).to(device)

            # Predict
            output = model(T_input)

            # Extract mean and std (denormalize)
            pred_mean = output[0, 0].cpu().numpy() * S_std + S_mean
            pred_std = output[0, 1].cpu().numpy() * S_std

            predictions.append(pred_mean)
            uncertainties.append(pred_std)

            print(f"  Day {day}: done")

    # Ground truth (from ORIGINAL ocean-only data)
    truths = [S_T_true[day] for day in sample_days]

    # -------------------------------------------------------------------------
    # Plot 1: Predictions vs Truth (maps)
    # -------------------------------------------------------------------------
    print("\nGenerating plots...")

    n_samples = len(sample_days)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, day in enumerate(sample_days):
        truth = truths[i].copy()
        pred = predictions[i].copy()
        unc = uncertainties[i].copy()

        # Mask land
        truth[~ocean_mask] = np.nan
        pred[~ocean_mask] = np.nan
        unc[~ocean_mask] = np.nan

        # Calculate error
        error = pred - truth

        # Find common scale
        vmin = np.nanpercentile(truth, 2)
        vmax = np.nanpercentile(truth, 98)

        # Plot truth
        im0 = axes[i, 0].imshow(truth, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Day {day}: Truth (S_T)')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        # Plot prediction
        im1 = axes[i, 1].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Day {day}: Prediction')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        # Plot error
        err_max = np.nanpercentile(np.abs(error), 95)
        im2 = axes[i, 2].imshow(error, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
        axes[i, 2].set_title(f'Day {day}: Error (Pred - Truth)')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        # Plot uncertainty
        im3 = axes[i, 3].imshow(unc, cmap='Oranges')
        axes[i, 3].set_title(f'Day {day}: Uncertainty (σ)')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

    plt.suptitle(f'Physics-Filled Model ({FILLED_VAR_NAME})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'predictions_maps_{FILLED_VAR_NAME}.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved predictions_maps_{FILLED_VAR_NAME}.png")

    # -------------------------------------------------------------------------
    # Plot 2: Scatter plot (Prediction vs Truth)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Combine all predictions and truths
    all_pred = np.concatenate([p[ocean_mask].flatten() for p in predictions])
    all_truth = np.concatenate([t[ocean_mask].flatten() for t in truths])

    # Remove NaN
    valid = np.isfinite(all_pred) & np.isfinite(all_truth)
    all_pred = all_pred[valid]
    all_truth = all_truth[valid]

    # Compute R²
    ss_res = np.sum((all_truth - all_pred) ** 2)
    ss_tot = np.sum((all_truth - np.mean(all_truth)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Compute correlation
    corr = np.corrcoef(all_pred, all_truth)[0, 1]

    # Compute RMSE
    rmse = np.sqrt(np.mean((all_pred - all_truth) ** 2))

    # Scatter plot (subsample for speed)
    n_points = len(all_pred)
    if n_points > 50000:
        idx = np.random.choice(n_points, 50000, replace=False)
        plot_pred = all_pred[idx]
        plot_truth = all_truth[idx]
    else:
        plot_pred = all_pred
        plot_truth = all_truth

    axes[0].scatter(plot_truth, plot_pred, alpha=0.1, s=1)

    # 1:1 line
    lims = [min(plot_truth.min(), plot_pred.min()),
            max(plot_truth.max(), plot_pred.max())]
    axes[0].plot(lims, lims, 'r--', label='1:1 line')

    axes[0].set_xlabel('Truth (S_T)')
    axes[0].set_ylabel('Prediction')
    axes[0].set_title(f'Prediction vs Truth ({FILLED_VAR_NAME})\nR² = {r2:.4f}, Corr = {corr:.4f}, RMSE = {rmse:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of errors
    errors = all_pred - all_truth
    axes[1].hist(errors, bins=100, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', label='Zero error')
    axes[1].axvline(np.mean(errors), color='g', linestyle='-', label=f'Mean: {np.mean(errors):.4f}')
    axes[1].set_xlabel('Error (Prediction - Truth)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Error Distribution\nStd = {np.std(errors):.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'predictions_scatter_{FILLED_VAR_NAME}.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved predictions_scatter_{FILLED_VAR_NAME}.png")

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"SUMMARY ({FILLED_VAR_NAME})")
    print("=" * 60)
    print(f"  R² Score:     {r2:.4f}")
    print(f"  Correlation:  {corr:.4f}")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  Mean Error:   {np.mean(errors):.4f}")
    print(f"  Std Error:    {np.std(errors):.4f}")
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    print(f"  - predictions_maps_{FILLED_VAR_NAME}.png")
    print(f"  - predictions_scatter_{FILLED_VAR_NAME}.png")


if __name__ == '__main__':
    main()
