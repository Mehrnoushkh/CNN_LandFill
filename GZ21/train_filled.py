"""
train_filled.py

Training script for PHYSICS-FILLED temperature data.

Your filled data:
    - File: sst_filled_30day_2023.nc
    - Variables: sst_dirichlet, sst_neumann (both are filled)
    - Dimensions: time (30), depth (1), latitude (1800), longitude (3600)
    - No -999: Land is already filled with physics-based values!

Usage:
    python train_filled.py
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

torch.set_num_threads(32)

# =============================================================================
# *** YOUR SETTINGS ***
# =============================================================================

# Path to GZ21 repo
GZ21_PATH = '.'

# Physics-filled data file
FILLED_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/Neumann_DrichletBc/cplusplus/sst_filled_30day_2023.nc'

# Original data file (needed for ocean mask)
ORIGINAL_DATA_FILE = '/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc'

# Variable names
FILLED_VAR_NAME = 'sst_neumann'    # or 'sst_dirichlet'
ORIGINAL_VAR_NAME = 'analysed_sst'

# Original file fill value for land
FILL_VALUE = -999.0

# Settings
COARSEN_FACTOR = 4
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
TRAIN_FRACTION = 0.8

OUTPUT_DIR = './output_filled'
MAX_TIME_STEPS = 30  # Use all 30 days

# =============================================================================
# Import GZ21 model
# =============================================================================

sys.path.insert(0, GZ21_PATH)

try:
    from models.models1 import FullyCNN
    print("✓ Imported GZ21 FullyCNN")
except ImportError as e:
    print(f"✗ Could not import GZ21: {e}")
    print("  Run: git clone --branch forpy https://github.com/chzhangudel/GZ21.git")
    sys.exit(1)

# =============================================================================
# Helper classes
# =============================================================================

class SoftPlusTransform(nn.Module):
    """Ensures positive standard deviation."""
    def __init__(self, n_targets=1):
        super().__init__()
        self.n_targets = n_targets

    def forward(self, x):
        mean = x[:, :self.n_targets, :, :]
        log_std = x[:, self.n_targets:, :, :]
        std = F.softplus(log_std) + 1e-6
        return torch.cat([mean, std], dim=1)


class HeteroskedasticGaussianLoss(nn.Module):
    """Loss = 0.5 * log(σ²) + 0.5 * (target - μ)² / σ²"""
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
    """Dataset for temperature."""

    def __init__(self, T_input, S_T_target, ocean_mask):
        self.T = torch.tensor(T_input, dtype=torch.float32)
        self.S_T = torch.tensor(S_T_target, dtype=torch.float32)
        self.mask = torch.tensor(ocean_mask, dtype=torch.float32)

        # Normalize over ocean only
        valid = ocean_mask.astype(bool)
        self.T_mean = np.nanmean(T_input[:, valid])
        self.T_std = np.nanstd(T_input[:, valid]) + 1e-8
        self.S_mean = np.nanmean(S_T_target[:, valid])
        self.S_std = np.nanstd(S_T_target[:, valid]) + 1e-8

        print(f"  Normalization - T: mean={self.T_mean:.2f}, std={self.T_std:.2f}")
        print(f"  Normalization - S: mean={self.S_mean:.4f}, std={self.S_std:.4f}")

        self.T = (self.T - self.T_mean) / self.T_std
        self.S_T = (self.S_T - self.S_mean) / self.S_std

        # NaN → 0 (shouldn't be any, but just in case)
        self.T = torch.nan_to_num(self.T, nan=0.0)
        self.S_T = torch.nan_to_num(self.S_T, nan=0.0)

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, idx):
        return self.T[idx].unsqueeze(0), self.S_T[idx].unsqueeze(0)

# =============================================================================
# Main functions
# =============================================================================

def compute_subgrid_forcing(T_highres, factor):
    """S_T = coarsen(T²) - coarsen(T)²"""
    print(f"Computing forcing (factor={factor})...")

    n_times, ny, nx = T_highres.shape
    ny_trim = (ny // factor) * factor
    nx_trim = (nx // factor) * factor
    T = T_highres[:, :ny_trim, :nx_trim]

    T_reshaped = T.reshape(n_times, ny_trim // factor, factor, nx_trim // factor, factor)
    T_coarse = np.nanmean(T_reshaped, axis=(2, 4))
    T_sq_coarse = np.nanmean(T_reshaped ** 2, axis=(2, 4))
    S_T = T_sq_coarse - T_coarse ** 2

    print(f"  Input: {T_highres.shape} → Coarse: {T_coarse.shape}")
    return T_coarse, S_T


def main():
    print("=" * 60)
    print("PHYSICS-FILL TRAINING")
    print(f"Using: {FILLED_VAR_NAME}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Load FILLED data (your physics-based fill)
    # =========================================================================
    print(f"\n{'='*60}")
    print("LOADING FILLED DATA")
    print(f"{'='*60}")
    print(f"File: {FILLED_DATA_FILE}")
    
    ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
    print(ds_filled)
    
    # Get filled temperature - NOTE: squeeze out depth dimension!
    T_filled = ds_filled[FILLED_VAR_NAME].values  # Shape: (time, depth, lat, lon)
    T_filled = T_filled[:, 0, :, :]               # Shape: (time, lat, lon) - remove depth
    T_filled = T_filled[:MAX_TIME_STEPS, :, :]
    
    print(f"\nFilled SST shape: {T_filled.shape}")
    print(f"  → {T_filled.shape[0]} time steps, {T_filled.shape[1]}x{T_filled.shape[2]} grid")
    print(f"SST range: [{np.nanmin(T_filled):.2f}, {np.nanmax(T_filled):.2f}]")

    # =========================================================================
    # Load ORIGINAL data (ONLY for ocean mask)
    # =========================================================================
    print(f"\n{'='*60}")
    print("LOADING ORIGINAL DATA (only for ocean mask)")
    print(f"{'='*60}")
    print(f"File: {ORIGINAL_DATA_FILE}")
    
    ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
    T_orig = ds_orig[ORIGINAL_VAR_NAME].values
    T_orig = T_orig[:MAX_TIME_STEPS, :, :]
    
    # Create land mask from original data
    land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
    print(f"Land fraction: {land_mask.mean():.1%}")
    
    # Convert original to NaN over land (just to get ocean mask after coarsening)
    T_orig_with_nan = T_orig.copy().astype(float)
    T_orig_with_nan[land_mask] = np.nan

    # =========================================================================
    # Compute subgrid forcing (ALL from FILLED file)
    # =========================================================================
    print(f"\n{'='*60}")
    print("COMPUTING SUBGRID FORCING")
    print(f"{'='*60}")
    
    # For FILLED data - compute BOTH input and target from filled file
    T_coarse_filled, S_T_filled = compute_subgrid_forcing(T_filled, COARSEN_FACTOR)
    
    # For ocean mask only - coarsen the original (with NaN) to get ocean points
    T_coarse_orig, _ = compute_subgrid_forcing(T_orig_with_nan, COARSEN_FACTOR)
    ocean_mask = np.isfinite(T_coarse_orig[0])
    
    print(f"\nMasks:")
    print(f"  Total coarse grid points: {ocean_mask.size}")
    print(f"  Ocean points: {ocean_mask.sum()}")
    print(f"  Land points: {(~ocean_mask).sum()}")

    # Target S_T comes from FILLED file (evaluated only over ocean)
    S_T_target = S_T_filled.copy()
    S_T_target[:, ~ocean_mask] = 0.0  # Zero out land for loss masking

    print(f"\nS_T range (ocean): [{np.nanmin(S_T_target[S_T_target>0]):.2f}, {np.nanmax(S_T_target):.2f}]")

    # =========================================================================
    # Create dataset
    # =========================================================================
    print(f"\n{'='*60}")
    print("CREATING DATASET")
    print(f"{'='*60}")
    
    # INPUT: coarsened filled temperature
    # TARGET: S_T computed from original ocean-only data
    dataset = TemperatureDataset(T_coarse_filled, S_T_target, ocean_mask)

    # Split
    n = len(dataset)
    train_size = int(TRAIN_FRACTION * n)
    val_size = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    
    print(f"  Total samples: {n}")
    print(f"  Train: {train_size}, Val: {val_size}")

    # =========================================================================
    # Create model
    # =========================================================================
    print(f"\n{'='*60}")
    print("CREATING MODEL")
    print(f"{'='*60}")
    
    model = FullyCNN(n_in_channels=1, n_out_channels=2, padding='same')
    model.final_transformation = SoftPlusTransform(n_targets=1)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: FullyCNN (from GZ21)")
    print(f"  Parameters: {n_params:,}")

    # Mask tensor
    mask_tensor = torch.tensor(ocean_mask, dtype=torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # Optimizer & loss
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = HeteroskedasticGaussianLoss()

    # =========================================================================
    # Training loop
    # =========================================================================
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()

    best_loss = float('inf')
    best_state = None
    history = {'train': [], 'val': []}

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
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y, mask_tensor)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

    # =========================================================================
    # Save
    # =========================================================================
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    torch.save({
        'model_state_dict': best_state,
        'T_mean': dataset.T_mean,
        'T_std': dataset.T_std,
        'S_mean': dataset.S_mean,
        'S_std': dataset.S_std,
        'fill_method': FILLED_VAR_NAME,
    }, os.path.join(OUTPUT_DIR, f'model_{FILLED_VAR_NAME}.pth'))
    print(f"  ✓ Saved model_{FILLED_VAR_NAME}.pth")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training Loss ({FILLED_VAR_NAME})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f'loss_{FILLED_VAR_NAME}.png'), dpi=150)
    plt.close()
    print(f"  ✓ Saved loss_{FILLED_VAR_NAME}.png")

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_loss:.4f}")
    print(f"  Results in: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
