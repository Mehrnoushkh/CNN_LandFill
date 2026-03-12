import numpy as np
import xarray as xr
import torch
import sys
import os
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================

GZ21_PATH = '.'
sys.path.insert(0, GZ21_PATH)
from models.models1 import FullyCNN
import torch.nn as nn
import torch.nn.functional as F

ORIGINAL_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc'
FILLED_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/Neumann_DrichletBc/cplusplus/fordays/sst_filled_365days_2023.nc'

MODEL_ZERO_FILE = './output_comparison_1year/model_zero_fill.pth'
MODEL_PHYSICS_FILE = './output_comparison_1year/model_physics_fill.pth'

OUTPUT_DIR = './output_comparison_1year'

ORIGINAL_VAR_NAME = 'analysed_sst'
FILLED_VAR_NAME = 'sst_neumann'
FILL_VALUE = -999.0
COARSEN_FACTOR = 4
MAX_TIME_STEPS = 364

# Pacific box (far from land) - in COARSE grid coordinates
# Roughly: 180°W to 140°W, 20°S to 20°N (middle of Pacific)
# Adjust these based on your coarse grid!
# After coarsening 4×, your grid is (450, 900) for 0.4° resolution

# For 0.4° resolution global grid:
# Longitude: 0 to 360° → 0 to 900 indices
# Latitude: -90 to 90° → 0 to 450 indices

# Pacific box (adjust as needed):
# Lon: 180° to 220° (middle Pacific) → indices ~450 to 550
# Lat: -20° to 20° (equatorial) → indices ~175 to 275

#PACIFIC_BOX = {
#    'lat_min': 175,  # ~-20° 
#    'lat_max': 275,  # ~+20°
#    'lon_min': 450,  # ~180°
#    'lon_max': 550,  # ~220°
#}
PACIFIC_BOX= {
    'lat_min': 550,   # -35.0° S
    'lat_max': 600,   # -30.0° S
    'lon_min': 2300,  # 230.0° E (or -130° W)
    'lon_max': 2350,  # 235.0° E (or -125° W)
}
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
            T_input = (T_coarse[t] - T_mean) / T_std
            T_input = torch.tensor(T_input, dtype=torch.float32)
            T_input = T_input.unsqueeze(0).unsqueeze(0).to(device)
            
            output = model(T_input)
            pred = output[0, 0].cpu().numpy() * S_std + S_mean
            predictions.append(pred)
    
    return np.array(predictions)


def compute_r2(pred, truth, mask):
    """Compute R² over masked region."""
    p = pred[:, mask].flatten()
    t = truth[:, mask].flatten()
    valid = np.isfinite(p) & np.isfinite(t)
    p, t = p[valid], t[valid]
    
    if len(p) == 0:
        return np.nan
    
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def main():
    print("=" * 60)
    print("PACIFIC BOX COMPARISON")
    print("Testing R² far from land")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # Load models
    # =========================================================================
    print("\nLoading models...")
    model_zero, ckpt_zero = load_model(MODEL_ZERO_FILE, device)
    model_physics, ckpt_physics = load_model(MODEL_PHYSICS_FILE, device)
    print("  ✓ Models loaded")

    # =========================================================================
    # Load data
    # =========================================================================
    print("\nLoading data...")
    
    ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
    T_orig = ds_orig[ORIGINAL_VAR_NAME].values[:MAX_TIME_STEPS, :, :]
    
    land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
    T_orig_with_nan = T_orig.copy().astype(float)
    T_orig_with_nan[land_mask] = np.nan
    T_zero_filled = np.nan_to_num(T_orig_with_nan, nan=0.0)
    
    ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
    T_physics_filled = ds_filled[FILLED_VAR_NAME].values[:MAX_TIME_STEPS, 0, :, :]
    
    print(f"  Original shape: {T_orig.shape}")

    # =========================================================================
    # Compute coarse data
    # =========================================================================
    print("\nComputing coarse data...")
    
    T_coarse_zero, _ = compute_subgrid_forcing(T_zero_filled, COARSEN_FACTOR)
    T_coarse_physics, _ = compute_subgrid_forcing(T_physics_filled, COARSEN_FACTOR)
    T_coarse_orig, S_T_true = compute_subgrid_forcing(T_orig_with_nan, COARSEN_FACTOR)
    
    ocean_mask = np.isfinite(T_coarse_orig[0])
    
    print(f"  Coarse shape: {T_coarse_zero.shape}")
    print(f"  Ocean points: {ocean_mask.sum()}")

    # =========================================================================
    # Create Pacific box mask
    # =========================================================================
    print("\nCreating Pacific box mask...")
    
    ny, nx = ocean_mask.shape
    pacific_mask = np.zeros((ny, nx), dtype=bool)
    
    lat_min = PACIFIC_BOX['lat_min']
    lat_max = PACIFIC_BOX['lat_max']
    lon_min = PACIFIC_BOX['lon_min']
    lon_max = PACIFIC_BOX['lon_max']
    
    pacific_mask[lat_min:lat_max, lon_min:lon_max] = True
    
    # Only ocean points in Pacific box
    pacific_ocean_mask = pacific_mask & ocean_mask
    
    print(f"  Pacific box: lat[{lat_min}:{lat_max}], lon[{lon_min}:{lon_max}]")
    print(f"  Pacific ocean points: {pacific_ocean_mask.sum()}")
    
    # Check there's no land in the box
    land_in_box = pacific_mask & ~ocean_mask
    print(f"  Land points in box: {land_in_box.sum()}")
    
    if land_in_box.sum() > 0:
        print("  WARNING: Box contains some land! Adjust coordinates.")

    # =========================================================================
    # Make predictions
    # =========================================================================
    print("\nMaking predictions...")
    
    pred_zero = predict(model_zero, T_coarse_zero,
                        ckpt_zero['T_mean'], ckpt_zero['T_std'],
                        ckpt_zero['S_mean'], ckpt_zero['S_std'], device)
    
    pred_physics = predict(model_physics, T_coarse_physics,
                           ckpt_physics['T_mean'], ckpt_physics['T_std'],
                           ckpt_physics['S_mean'], ckpt_physics['S_std'], device)
    
    print("  ✓ Predictions done")

    # =========================================================================
    # Compute R² for different regions
    # =========================================================================
    print("\n" + "=" * 60)
    print("R² COMPARISON BY REGION")
    print("=" * 60)
    
    # Global ocean
    r2_zero_global = compute_r2(pred_zero, S_T_true, ocean_mask)
    r2_physics_global = compute_r2(pred_physics, S_T_true, ocean_mask)
    
    # Pacific box only
    r2_zero_pacific = compute_r2(pred_zero, S_T_true, pacific_ocean_mask)
    r2_physics_pacific = compute_r2(pred_physics, S_T_true, pacific_ocean_mask)
    
    print(f"\n{'Region':<30} {'Zero-fill R²':>15} {'Physics-fill R²':>15} {'Difference':>12}")
    print("-" * 75)
    print(f"{'Global Ocean':<30} {r2_zero_global:>15.4f} {r2_physics_global:>15.4f} {r2_physics_global - r2_zero_global:>12.4f}")
    print(f"{'Pacific Box (far from land)':<30} {r2_zero_pacific:>15.4f} {r2_physics_pacific:>15.4f} {r2_physics_pacific - r2_zero_pacific:>12.4f}")

    # =========================================================================
    # Plot
    # =========================================================================
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Bar chart comparison
    regions = ['Global\nOcean', 'Pacific Box\n(far from land)']
    zero_r2 = [r2_zero_global, r2_zero_pacific]
    physics_r2 = [r2_physics_global, r2_physics_pacific]
    
    x = np.arange(len(regions))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, zero_r2, width, label='Zero-fill', color='steelblue')
    bars2 = axes[0].bar(x + width/2, physics_r2, width, label='Physics-fill', color='coral')
    
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² Comparison: Global vs Pacific Box')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regions)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, max(max(zero_r2), max(physics_r2)) * 1.2)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    # Right: Show Pacific box location
    display_mask = np.zeros_like(ocean_mask, dtype=float)
    display_mask[~ocean_mask] = np.nan  # Land
    display_mask[ocean_mask] = 0        # Ocean
    display_mask[pacific_ocean_mask] = 1  # Pacific box
    
    im = axes[1].imshow(display_mask, origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    axes[1].set_title('Pacific Box Location (red)')
    axes[1].set_xlabel('Longitude index')
    axes[1].set_ylabel('Latitude index')
    
    # Draw box outline
    rect = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                          fill=False, edgecolor='black', linewidth=2)
    axes[1].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pacific_box_comparison.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ Saved pacific_box_comparison.png")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    diff_global = r2_physics_global - r2_zero_global
    diff_pacific = r2_physics_pacific - r2_zero_pacific
    
    print(f"\nGlobal improvement: {diff_global:.4f} ({diff_global/r2_zero_global*100:.1f}%)")
    print(f"Pacific box improvement: {diff_pacific:.4f} ({diff_pacific/r2_zero_pacific*100:.1f}%)")
    
    if abs(diff_pacific) < 0.01:
        print("\n→ Pacific box shows SIMILAR R² for both methods")
        print("  This confirms: difference is mainly at COASTS!")
    else:
        print("\n→ Pacific box still shows difference")
        print("  This is because models were trained on different data")


if __name__ == '__main__':
    main()
