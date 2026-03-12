import xarray as xr
import torch
import sys
import os
import numpy as np  # Added missing numpy import
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# SETTINGS
# =============================================================================

GZ21_PATH = '.'
sys.path.insert(0, GZ21_PATH)
from models.models1 import FullyCNN

ORIGINAL_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc'
FILLED_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/Neumann_DrichletBc/cplusplus/fordays/sst_filled_365days_2023.nc'

MODEL_ZERO_FILE = './output_comparison_1year/model_zero_fill.pth'
MODEL_PHYSICS_FILE = './output_comparison_1year/model_physics_fill.pth'

OUTPUT_DIR = './output_comparison_1year'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ORIGINAL_VAR_NAME = 'analysed_sst'
FILLED_VAR_NAME = 'sst_neumann'
FILL_VALUE = -999.0
COARSEN_FACTOR = 4
MAX_TIME_STEPS = 364

# Pacific box coordinates converted to COARSE grid (Divided by 4)
# High-res indices: Lat 550-600, Lon 2300-2350
# Updated based on your specific NetCDF indexing
PACIFIC_BOX = {
    'lat_min': 400 // COARSEN_FACTOR,  # -49.97° S
    'lat_max': 600 // COARSEN_FACTOR,  # -29.97° S
    'lon_min': 280 // COARSEN_FACTOR,  # -151.97° W
    'lon_max': 480 // COARSEN_FACTOR,  # -131.97° W
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
    # Using np.nanmean to handle land masks correctly
    T_coarse = np.nanmean(T_reshaped, axis=(2, 4))
    T_sq_coarse = np.nanmean(T_reshaped ** 2, axis=(2, 4))
    S_T = T_sq_coarse - T_coarse ** 2

    return T_coarse, S_T

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = FullyCNN(n_in_channels=1, n_out_channels=2, padding='same')
    model.final_transformation = SoftPlusTransform(n_targets=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint

def predict(model, T_coarse, T_mean, T_std, S_mean, S_std, device):
    predictions = []
    with torch.no_grad():
        for t in range(T_coarse.shape[0]):
            T_input = (T_coarse[t] - T_mean) / T_std
            T_input = torch.tensor(T_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(T_input)
            # Rescale back to physical units
            pred = output[0, 0].cpu().numpy() * S_std + S_mean
            predictions.append(pred)
    return np.array(predictions)

def compute_r2(pred, truth, mask):
    p = pred[:, mask].flatten()
    t = truth[:, mask].flatten()
    valid = np.isfinite(p) & np.isfinite(t)
    p, t = p[valid], t[valid]
    if len(p) == 0: return np.nan
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
    T_orig = ds_orig[ORIGINAL_VAR_NAME].values[:MAX_TIME_STEPS, :, :]
    
    # Masking logic
    land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
    T_orig_nan = T_orig.copy().astype(float)
    T_orig_nan[land_mask] = np.nan
    
    T_zero_filled = np.nan_to_num(T_orig_nan, nan=0.0)
    
    ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
    # Handle possible 4D shape (time, depth, lat, lon)
    T_phys_raw = ds_filled[FILLED_VAR_NAME].values[:MAX_TIME_STEPS]
    T_physics_filled = T_phys_raw[:, 0, :, :] if T_phys_raw.ndim == 4 else T_phys_raw

    # Coarsen
    T_coarse_zero, _ = compute_subgrid_forcing(T_zero_filled, COARSEN_FACTOR)
    T_coarse_physics, _ = compute_subgrid_forcing(T_physics_filled, COARSEN_FACTOR)
    T_coarse_orig, S_T_true = compute_subgrid_forcing(T_orig_nan, COARSEN_FACTOR)

    ocean_mask = np.isfinite(T_coarse_orig[0])

    # Pacific Mask
    ny, nx = ocean_mask.shape
    pacific_mask = np.zeros((ny, nx), dtype=bool)
    pacific_mask[PACIFIC_BOX['lat_min']:PACIFIC_BOX['lat_max'], 
                 PACIFIC_BOX['lon_min']:PACIFIC_BOX['lon_max']] = True
    pacific_ocean_mask = pacific_mask & ocean_mask

    # Run Models
    model_z, ckpt_z = load_model(MODEL_ZERO_FILE, device)
    pred_zero = predict(model_z, T_coarse_zero, ckpt_z['T_mean'], ckpt_z['T_std'], ckpt_z['S_mean'], ckpt_z['S_std'], device)
    
    model_p, ckpt_p = load_model(MODEL_PHYSICS_FILE, device)
    pred_phys = predict(model_p, T_coarse_physics, ckpt_p['T_mean'], ckpt_p['T_std'], ckpt_p['S_mean'], ckpt_p['S_std'], device)

    # Metrics
    r2_z_glob = compute_r2(pred_zero, S_T_true, ocean_mask)
    r2_p_glob = compute_r2(pred_phys, S_T_true, ocean_mask)
    r2_z_pac = compute_r2(pred_zero, S_T_true, pacific_ocean_mask)
    r2_p_pac = compute_r2(pred_phys, S_T_true, pacific_ocean_mask)

    print(f"Global R2: Zero={r2_z_glob:.4f}, Physics={r2_p_glob:.4f}")
    print(f"Pacific R2: Zero={r2_z_pac:.4f}, Physics={r2_p_pac:.4f}")

    # Plotting code remains similar...
    # (Removed for brevity but ensure you use the updated PACIFIC_BOX indices)

if __name__ == '__main__':
    main()
