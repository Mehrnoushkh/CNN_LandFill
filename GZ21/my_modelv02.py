"""
train_single.py

Simple training script for ONE case: land (-999) → zero-filled.

Your data:
    - File: updated_data/res005/cs3_glo_2023.nc (or similar)
    - Variable: analysed_sst
    - Dimensions: time (365), latitude (1800), longitude (3600)
    - Fill value: -999 (land)
    - Resolution: 0.1° → coarsen 4x → 0.4°

Usage:
    python train_single.py
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
import torch
torch.set_num_threads(32)  # Use 32 cores for matrix operations
NUM_WORKERS = 0         # Use 16 cores for data loading
# =============================================================================
# *** YOUR SETTINGS (already configured for your file) ***
# =============================================================================

# Path to GZ21 repo
GZ21_PATH = '.'

# Your data file

DATA_FILE = './../updated_data/res005/data2023_01deg.nc'

# From your ncdump output:
T_VAR_NAME ='analysed_sst'
TIME_DIM = 'time'
LAT_DIM = 'latitude'
LON_DIM = 'longitude'
FILL_VALUE = -999.0  # Land points have this value

# Settings
COARSEN_FACTOR = 4   # 0.1° → 0.4°1
EPOCHS = 100
BATCH_SIZE =32 #8
LEARNING_RATE = 5e-4
TRAIN_FRACTION = 0.8

OUTPUT_DIR = 'output_single' #'./output_filled'
MAX_TIME_STEPS = 30
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
    """Simple dataset for temperature."""
    
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
        
        self.T = (self.T - self.T_mean) / self.T_std
        self.S_T = (self.S_T - self.S_mean) / self.S_std
        
        # NaN → 0
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
    print("SINGLE CASE TRAINING: Zero-fill (land → 0)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"\nLoading {DATA_FILE}...")
    ds = xr.open_dataset(DATA_FILE,decode_times=False)
    print(ds)
    T = ds[T_VAR_NAME].values
    T = T[:MAX_TIME_STEPS, :, :]  # Add this line - only first 100 days
    print(f"\nSST shape: {T.shape}")
    print(f"  → {T.shape[0]} time steps, {T.shape[1]}x{T.shape[2]} grid")
    
    # Handle fill value (-999 = land)
    land_mask = (T == FILL_VALUE)
    print(f"Land fraction: {land_mask.mean():.1%}")
    
    # Convert -999 to NaN for easier handling
    T_with_nan = T.copy()
    T_with_nan[land_mask] = np.nan
    print(f"SST range (ocean): [{np.nanmin(T_with_nan):.2f}, {np.nanmax(T_with_nan):.2f}]")
    
    # Zero-fill for training input
    T_zero = np.nan_to_num(T_with_nan, nan=0.0)
    
    # Compute forcing
    T_coarse, S_T = compute_subgrid_forcing(T_zero, COARSEN_FACTOR)
    
    # Create ocean mask (where original data was valid, not -999)
    T_orig_coarse, _ = compute_subgrid_forcing(T_with_nan, COARSEN_FACTOR)
    ocean_mask = np.isfinite(T_orig_coarse[0])
    print(f"Ocean points (coarse): {ocean_mask.sum()}")
    
    # Mask the forcing (only compute loss over ocean)
    S_T[:, ~ocean_mask] = 0.0
    
    # Create dataset
    dataset = TemperatureDataset(T_coarse, S_T, ocean_mask)
    
    # Split
    n = len(dataset)
    train_size = int(TRAIN_FRACTION * n)
    val_size = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    
   # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
   # val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    #train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
     #                     num_workers=NUM_WORKERS)
    #val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                        num_workers=0)
    print(f"Train: {train_size}, Val: {val_size}")
    
    # Create model
    model = FullyCNN(n_in_channels=1, n_out_channels=2, padding='same')
    model.final_transformation = SoftPlusTransform(n_targets=1)
    model = model.to(device)
    
    # Mask tensor
    mask_tensor = torch.tensor(ocean_mask, dtype=torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    # Optimizer & loss
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = HeteroskedasticGaussianLoss()
    
    # Training loop
    print("\nTraining...")
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
    
    # Save
    print("\nSaving model...")
    torch.save({
        'model_state_dict': best_state,
        'T_mean': dataset.T_mean,
        'T_std': dataset.T_std,
        'S_mean': dataset.S_mean,
        'S_std': dataset.S_std,
    }, os.path.join(OUTPUT_DIR, 'model_300days.pth'))
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_300days.png'), dpi=150)
    plt.close()
    
    print(f"\n✓ Done! Results in {OUTPUT_DIR}/")
    print(f"  Best val loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
