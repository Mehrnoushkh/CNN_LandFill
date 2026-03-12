"""
compare_methods.py

FAIR COMPARISON: Both methods predict the SAME ocean-only S_T target.

- Zero-fill input:    T with 0 over land
- Physics-fill input: T with physics values over land
- Target (BOTH):      S_T computed from original ocean data only

Usage:
    python compare_methods.py
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
FILLED_VAR_NAME = 'sst_neumann'  # or 'sst_dirichlet'
FILL_VALUE = -999.0

# Settings
COARSEN_FACTOR = 4
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
TRAIN_FRACTION = 0.8
MAX_TIME_STEPS = 364

OUTPUT_DIR = './output_comparison_1year'

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


def evaluate_model(model, data_loader, ocean_mask, S_mean, S_std, device):
    """Evaluate model and return predictions, truths, and metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # Denormalize
            pred = out[:, 0:1, :, :].cpu().numpy() * S_std + S_mean
            target = y.cpu().numpy() * S_std + S_mean
            all_preds.append(pred)
            all_targets.append(target)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics over ocean only
    p = preds[:, 0, ocean_mask].flatten()
    t = targets[:, 0, ocean_mask].flatten()

    valid = np.isfinite(p) & np.isfinite(t)
    p, t = p[valid], t[valid]

    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    rmse = np.sqrt(np.mean((p - t) ** 2))
    corr = np.corrcoef(p, t)[0, 1]

    return {
        'r2': r2,
        'rmse': rmse,
        'corr': corr,
        'predictions': preds,
        'targets': targets
    }


def main():
    print("=" * 60)
    print("FAIR COMPARISON: Zero-fill vs Physics-fill")
    print("Both predict the SAME ocean-only S_T target")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Load ORIGINAL data
    # =========================================================================
    print(f"\n{'='*60}")
    print("LOADING ORIGINAL DATA")
    print(f"{'='*60}")

    ds_orig = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
    T_orig = ds_orig[ORIGINAL_VAR_NAME].values[:MAX_TIME_STEPS, :, :]

    print(f"Shape: {T_orig.shape}")

    # Create masks
    land_mask = (T_orig == FILL_VALUE) | np.isnan(T_orig)
    print(f"Land fraction: {land_mask.mean():.1%}")

    # T with NaN over land (for computing TRUE S_T)
    T_orig_with_nan = T_orig.copy().astype(float)
    T_orig_with_nan[land_mask] = np.nan

    # T with 0 over land (zero-fill INPUT)
    T_zero_filled = np.nan_to_num(T_orig_with_nan, nan=0.0)

    # =========================================================================
    # Load FILLED data (physics-fill INPUT)
    # =========================================================================
    print(f"\n{'='*60}")
    print("LOADING FILLED DATA")
    print(f"{'='*60}")

    ds_filled = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
    T_physics_filled = ds_filled[FILLED_VAR_NAME].values[:MAX_TIME_STEPS, 0, :, :]  # Squeeze depth

    print(f"Shape: {T_physics_filled.shape}")

    # =========================================================================
    # Compute coarse data
    # =========================================================================
    print(f"\n{'='*60}")
    print("COMPUTING COARSE DATA AND TARGET")
    print(f"{'='*60}")

    # Coarse zero-filled (INPUT for method A)
    T_coarse_zero, _ = compute_subgrid_forcing(T_zero_filled, COARSEN_FACTOR)
    print(f"Zero-fill coarse shape: {T_coarse_zero.shape}")

    # Coarse physics-filled (INPUT for method B)
    T_coarse_physics, _ = compute_subgrid_forcing(T_physics_filled, COARSEN_FACTOR)
    print(f"Physics-fill coarse shape: {T_coarse_physics.shape}")

    # Coarse original with NaN (for OCEAN MASK and TRUE S_T TARGET)
    T_coarse_orig, S_T_target = compute_subgrid_forcing(T_orig_with_nan, COARSEN_FACTOR)
    ocean_mask = np.isfinite(T_coarse_orig[0])
    print(f"Ocean mask shape: {ocean_mask.shape}")
    print(f"Ocean points: {ocean_mask.sum()}")

    # THE SAME TARGET FOR BOTH METHODS!
    S_T_target_masked = S_T_target.copy()
    S_T_target_masked[:, ~ocean_mask] = 0.0

    print(f"\nTarget S_T (ocean-only) range: [{np.nanmin(S_T_target):.4f}, {np.nanmax(S_T_target):.4f}]")

    # =========================================================================
    # Create datasets (SAME TARGET, different inputs)
    # =========================================================================
    print(f"\n{'='*60}")
    print("CREATING DATASETS")
    print(f"{'='*60}")

    dataset_zero = TemperatureDataset(T_coarse_zero, S_T_target_masked, ocean_mask)
    dataset_physics = TemperatureDataset(T_coarse_physics, S_T_target_masked, ocean_mask)

    # Same split for both (fair comparison)
    n = len(dataset_zero)
    train_size = int(TRAIN_FRACTION * n)
    val_size = n - train_size
    generator = torch.Generator().manual_seed(42)

    train_zero, val_zero = random_split(dataset_zero, [train_size, val_size], generator=generator)
    generator = torch.Generator().manual_seed(42)  # Reset for same split
    train_physics, val_physics = random_split(dataset_physics, [train_size, val_size], generator=generator)

    train_loader_zero = DataLoader(train_zero, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_zero = DataLoader(val_zero, batch_size=BATCH_SIZE, num_workers=0)

    train_loader_physics = DataLoader(train_physics, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_physics = DataLoader(val_physics, batch_size=BATCH_SIZE, num_workers=0)

    print(f"Train: {train_size}, Val: {val_size}")

    # Mask tensor
    mask_tensor = torch.tensor(ocean_mask, dtype=torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # =========================================================================
    # Train BOTH models
    # =========================================================================

    # Method A: Zero-fill
    model_zero, history_zero, best_loss_zero = train_model(
        train_loader_zero, val_loader_zero, mask_tensor, device, "ZERO-FILL"
    )

    # Method B: Physics-fill
    model_physics, history_physics, best_loss_physics = train_model(
        train_loader_physics, val_loader_physics, mask_tensor, device, "PHYSICS-FILL (Neumann)"
    )

    # =========================================================================
    # Evaluate BOTH models
    # =========================================================================
    print(f"\n{'='*60}")
    print("EVALUATING MODELS")
    print(f"{'='*60}")

    results_zero = evaluate_model(
        model_zero, val_loader_zero, ocean_mask,
        dataset_zero.S_mean, dataset_zero.S_std, device
    )

    results_physics = evaluate_model(
        model_physics, val_loader_physics, ocean_mask,
        dataset_physics.S_mean, dataset_physics.S_std, device
    )

    # =========================================================================
    # Print comparison
    # =========================================================================
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Method':<25} {'R²':>10} {'Corr':>10} {'RMSE':>12}")
    print("-" * 60)
    print(f"{'Zero-fill':<25} {results_zero['r2']:>10.4f} {results_zero['corr']:>10.4f} {results_zero['rmse']:>12.4f}")
    print(f"{'Physics-fill (Neumann)':<25} {results_physics['r2']:>10.4f} {results_physics['corr']:>10.4f} {results_physics['rmse']:>12.4f}")

    # =========================================================================
    # Save models
    # =========================================================================
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print(f"{'='*60}")

    torch.save({
        'model_state_dict': model_zero.state_dict(),
        'T_mean': dataset_zero.T_mean,
        'T_std': dataset_zero.T_std,
        'S_mean': dataset_zero.S_mean,
        'S_std': dataset_zero.S_std,
    }, os.path.join(OUTPUT_DIR, 'model_zero_fill.pth'))

    torch.save({
        'model_state_dict': model_physics.state_dict(),
        'T_mean': dataset_physics.T_mean,
        'T_std': dataset_physics.T_std,
        'S_mean': dataset_physics.S_mean,
        'S_std': dataset_physics.S_std,
    }, os.path.join(OUTPUT_DIR, 'model_physics_fill.pth'))

    print("  ✓ Saved model_zero_fill.pth")
    print("  ✓ Saved model_physics_fill.pth")

    # =========================================================================
    # Generate comparison plots
    # =========================================================================
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")

    # Plot 1: Training curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_zero['train'], label='Train', color='blue')
    axes[0].plot(history_zero['val'], label='Val', color='orange')
    axes[0].set_title('Zero-fill Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_physics['train'], label='Train', color='blue')
    axes[1].plot(history_physics['val'], label='Val', color='orange')
    axes[1].set_title('Physics-fill Training')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Loss Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_comparison.png'), dpi=150)
    plt.close()
    print("  ✓ Saved training_comparison.png")

    # Plot 2: R² comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['Zero-fill', 'Physics-fill\n(Neumann)']
    r2_values = [results_zero['r2'], results_physics['r2']]
    colors = ['steelblue', 'coral']

    bars = ax.bar(methods, r2_values, color=colors, edgecolor='black')

    for bar, r2 in zip(bars, r2_values):
        ax.annotate(f'{r2:.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Fair Comparison: Same Ocean-Only Target\n(Higher is Better)', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_comparison.png'), dpi=150)
    plt.close()
    print("  ✓ Saved r2_comparison.png")

    # Plot 3: Scatter plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, results, title in [
        (axes[0], results_zero, 'Zero-fill'),
        (axes[1], results_physics, 'Physics-fill')
    ]:
        preds = results['predictions'][:, 0, ocean_mask].flatten()
        targets = results['targets'][:, 0, ocean_mask].flatten()

        valid = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[valid], targets[valid]

        # Subsample
        if len(preds) > 30000:
            idx = np.random.choice(len(preds), 30000, replace=False)
            preds, targets = preds[idx], targets[idx]

        ax.scatter(targets, preds, alpha=0.2, s=1)
        lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
        ax.plot(lims, lims, 'r--', label='1:1 line')
        ax.set_xlabel('Truth (S_T)')
        ax.set_ylabel('Prediction')
        ax.set_title(f'{title}\nR² = {results["r2"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Prediction vs Truth (Ocean-Only Target)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_comparison.png'), dpi=150)
    plt.close()
    print("  ✓ Saved scatter_comparison.png")

    # Save results to JSON
    results_json = {
        'zero_fill': {
            'r2': float(results_zero['r2']),
            'rmse': float(results_zero['rmse']),
            'corr': float(results_zero['corr']),
        },
        'physics_fill': {
            'r2': float(results_physics['r2']),
            'rmse': float(results_physics['rmse']),
            'corr': float(results_physics['corr']),
        },
        'settings': {
            'days': MAX_TIME_STEPS,
            'epochs': EPOCHS,
            'coarsen_factor': COARSEN_FACTOR,
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    print("  ✓ Saved results.json")

    # =========================================================================
    # Final summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"""
    Both methods predict the SAME target: ocean-only S_T
    
    Results ({MAX_TIME_STEPS} days, {EPOCHS} epochs):
    
    ┌─────────────────────────┬────────────┬────────────┐
    │ Method                  │     R²     │    RMSE    │
    ├─────────────────────────┼────────────┼────────────┤
    │ Zero-fill               │  {results_zero['r2']:>8.4f}  │  {results_zero['rmse']:>8.4f}  │
    │ Physics-fill (Neumann)  │  {results_physics['r2']:>8.4f}  │  {results_physics['rmse']:>8.4f}  │
    └─────────────────────────┴────────────┴────────────┘
    
    Outputs saved to: {OUTPUT_DIR}/
    """)


if __name__ == '__main__':
    main()
