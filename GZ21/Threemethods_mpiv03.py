"""
Simple MPI: 10 seeds, 5 ranks
Run with: ibrun -np 5 python train_5seeds_mpi.py
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
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
EPOCHS = 50                # Reduced from 100
BATCH_SIZE = 32            # Reduced from 64 to save memory
LEARNING_RATE = 5e-4
TRAIN_FRACTION = 0.8
MAX_TIME_STEPS = 364
EARLY_STOP_PATIENCE = 10   # Stop if no improvement

OUTPUT_DIR = './output_10seeds'
PREPROCESSED_FILE = './preprocessed_data.npz'

# 10 SEEDS
SEEDS = [42, 123, 456, 789, 1011, 12, 124, 458, 8, 65]

# Each rank uses more threads since fewer ranks
torch.set_num_threads(16)

# =============================================================================
# Helpers
# =============================================================================

def log(msg):
    """Print with rank prefix and flush."""
    print(f"[Rank {rank}] {msg}", flush=True)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.insert(0, GZ21_PATH)
from models.models1 import FullyCNN

# =============================================================================
# Model classes
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
# Data functions
# =============================================================================

def compute_subgrid_forcing(T_highres, factor):
    n_times, ny, nx = T_highres.shape
    ny_trim = (ny // factor) * factor
    nx_trim = (nx // factor) * factor
    ny_c, nx_c = ny_trim // factor, nx_trim // factor

    T_coarse = np.zeros((n_times, ny_c, nx_c), dtype=np.float32)
    S_T = np.zeros((n_times, ny_c, nx_c), dtype=np.float32)

    for t in range(n_times):
        T_t = T_highres[t, :ny_trim, :nx_trim].reshape(ny_c, factor, nx_c, factor)
        T_coarse[t] = np.nanmean(T_t, axis=(1, 3))
        S_T[t] = np.nanmean(T_t**2, axis=(1, 3)) - T_coarse[t]**2

    return T_coarse, S_T


def create_replicate_fill(T_with_nan, land_mask_3d):
    T_rep = T_with_nan.copy()
    land_2d = land_mask_3d[0]
    _, indices = distance_transform_edt(land_2d, return_indices=True)
    rows, cols = indices[0][land_2d], indices[1][land_2d]
    for t in range(T_with_nan.shape[0]):
        T_rep[t, land_2d] = T_with_nan[t, rows, cols]
    return np.nan_to_num(T_rep, nan=288.0)


def broadcast_array(arr, root=0):
    """Efficient numpy array broadcast."""
    if rank == root:
        shape, dtype = arr.shape, arr.dtype
    else:
        shape, dtype = None, None
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    if rank != root:
        arr = np.empty(shape, dtype=dtype)
    comm.Bcast(arr, root=root)
    return arr

# =============================================================================
# Training functions
# =============================================================================

def train_model(train_loader, val_loader, mask_tensor, device, method_name):
    model = FullyCNN(n_in_channels=1, n_out_channels=2, padding='same')
    model.final_transformation = SoftPlusTransform()
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = HeteroskedasticGaussianLoss()

    best_loss, best_state, patience = float('inf'), None, 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y, mask_tensor)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y, mask_tensor).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        # Progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            log(f"  {method_name} epoch {epoch+1}: val_loss={val_loss:.4f}")

        # Early stopping
        if patience >= EARLY_STOP_PATIENCE:
            log(f"  {method_name} early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, loader, ocean_mask, coastal_mask, S_mean, S_std, device):
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.append(out[:, 0:1].cpu().numpy() * S_std + S_mean)
            targets.append(y.numpy() * S_std + S_mean)
    
    preds = np.concatenate(preds)[:, 0]
    targets = np.concatenate(targets)[:, 0]

    def r2(mask):
        p, t = preds[:, mask].flatten(), targets[:, mask].flatten()
        valid = np.isfinite(p) & np.isfinite(t)
        p, t = p[valid], t[valid]
        ss_res = np.sum((t - p)**2)
        ss_tot = np.sum((t - t.mean())**2)
        return 1 - ss_res / (ss_tot + 1e-8)

    open_ocean = ocean_mask & ~coastal_mask
    return {'overall': r2(ocean_mask), 'coastal': r2(coastal_mask), 'open_ocean': r2(open_ocean)}, preds, targets


def get_full_predictions(model, T_input, S_T_target, ocean_mask, device):
    """Get time-averaged predictions. Memory efficient - no full dataset copy."""
    model.eval()
    
    n_times, ny, nx = T_input.shape
    
    # Compute normalization stats
    valid = ocean_mask.astype(bool)
    T_mean = np.nanmean(T_input[:, valid])
    T_std = np.nanstd(T_input[:, valid]) + 1e-8
    S_mean = np.nanmean(S_T_target[:, valid])
    S_std = np.nanstd(S_T_target[:, valid]) + 1e-8
    
    # Accumulate predictions
    pred_sum = np.zeros((ny, nx), dtype=np.float64)
    
    # Process one time step at a time to minimize memory
    with torch.no_grad():
        for t in range(n_times):
            # Normalize input
            x = (T_input[t] - T_mean) / T_std
            x = np.nan_to_num(x, nan=0.0)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Forward pass
            out = model(x_tensor)
            pred = out[0, 0].cpu().numpy() * S_std + S_mean
            
            pred_sum += pred
            
            # Print progress every 100 steps
            if (t + 1) % 100 == 0:
                print(f"    Predictions: {t+1}/{n_times}", flush=True)
    
    return (pred_sum / n_times).astype(np.float32)

# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    
    # Distribute seeds among ranks (some ranks may handle multiple seeds)
    seeds_per_rank = len(SEEDS) // size
    remainder = len(SEEDS) % size
    
    if rank < remainder:
        start_idx = rank * (seeds_per_rank + 1)
        my_seeds = SEEDS[start_idx : start_idx + seeds_per_rank + 1]
    else:
        start_idx = remainder * (seeds_per_rank + 1) + (rank - remainder) * seeds_per_rank
        my_seeds = SEEDS[start_idx : start_idx + seeds_per_rank]
    
    if rank == 0:
        print("=" * 60, flush=True)
        print("10-SEED MPI TRAINING", flush=True)
        print(f"Ranks: {size}, Seeds: {SEEDS}", flush=True)
        print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}", flush=True)
        print("=" * 60, flush=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    comm.Barrier()
    log(f"My seeds: {my_seeds}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # =========================================================================
    # Load data (Rank 0 only, then broadcast)
    # =========================================================================
    
    if rank == 0:
        if os.path.exists(PREPROCESSED_FILE):
            log("Loading from cache...")
            data = np.load(PREPROCESSED_FILE)
            T_zero = data['T_coarse_zero']
            T_rep = data['T_coarse_replicate']
            T_lap = data['T_coarse_laplace']
            S_T = data['S_T_target_masked']
            ocean_mask = data['ocean_mask']
            coastal_mask = data['coastal_mask']
        else:
            log("Loading raw data...")
            ds = xr.open_dataset(ORIGINAL_DATA_FILE, decode_times=False)
            T_orig = ds[ORIGINAL_VAR_NAME].values[:MAX_TIME_STEPS].astype(np.float32)
            ds.close()

            land = (T_orig == FILL_VALUE) | np.isnan(T_orig)
            T_nan = T_orig.copy()
            T_nan[land] = np.nan

            log("Creating fills...")
            T_zero_raw = np.nan_to_num(T_nan, nan=0.0)
            T_rep_raw = create_replicate_fill(T_nan, land)
            
            ds = xr.open_dataset(FILLED_DATA_FILE, decode_times=False)
            T_lap_raw = ds[FILLED_VAR_NAME].values[:MAX_TIME_STEPS, 0].astype(np.float32)
            ds.close()

            log("Coarsening...")
            T_orig_c, S_T = compute_subgrid_forcing(T_nan, COARSEN_FACTOR)
            ocean_mask = np.isfinite(T_orig_c[0])
            coastal_mask = binary_dilation(~ocean_mask, iterations=10) & ocean_mask
            S_T[:, ~ocean_mask] = 0.0

            T_zero, _ = compute_subgrid_forcing(T_zero_raw, COARSEN_FACTOR)
            T_rep, _ = compute_subgrid_forcing(T_rep_raw, COARSEN_FACTOR)
            T_lap, _ = compute_subgrid_forcing(T_lap_raw, COARSEN_FACTOR)

            log("Saving cache...")
            np.savez(PREPROCESSED_FILE,
                     T_coarse_zero=T_zero, T_coarse_replicate=T_rep,
                     T_coarse_laplace=T_lap, S_T_target_masked=S_T,
                     ocean_mask=ocean_mask, coastal_mask=coastal_mask)
        
        log(f"Data shape: {T_zero.shape}")
    else:
        T_zero = T_rep = T_lap = S_T = ocean_mask = coastal_mask = None
    
    # Broadcast
    if rank == 0:
        log("Broadcasting...")
    
    T_zero = broadcast_array(T_zero)
    T_rep = broadcast_array(T_rep)
    T_lap = broadcast_array(T_lap)
    S_T = broadcast_array(S_T)
    ocean_mask = broadcast_array(ocean_mask)
    coastal_mask = broadcast_array(coastal_mask)
    
    comm.Barrier()
    log("Data received")
    
    mask_tensor = torch.tensor(ocean_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # =========================================================================
    # Train 3 methods FOR EACH SEED
    # =========================================================================
    
    import gc  # Garbage collection
    
    all_my_results = []  # Store results for all seeds this rank handles
    
    for my_seed in my_seeds:
        log(f"=== Starting seed {my_seed} ===")
        
        set_all_seeds(my_seed)
        results = {}
        spatial_preds = {}
        
        for name, T in [('zero_fill', T_zero), ('replicate_fill', T_rep), ('laplace_fill', T_lap)]:
            log(f"Training {name}...")
            t0 = time.time()
            
            ds = TemperatureDataset(T, S_T, ocean_mask)
            n = len(ds)
            train_n = int(TRAIN_FRACTION * n)
            
            gen = torch.Generator().manual_seed(my_seed)
            train_ds, val_ds = random_split(ds, [train_n, n - train_n], generator=gen)
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
            
            model = train_model(train_loader, val_loader, mask_tensor, device, name)
            r2, _, _ = evaluate_model(model, val_loader, ocean_mask, coastal_mask, ds.S_mean, ds.S_std, device)
            
            # Get spatial predictions (time-averaged)
            spatial_preds[name] = get_full_predictions(model, T, S_T, ocean_mask, device)
            
            results[name] = {k: float(v) for k, v in r2.items()}
            log(f"  {name} done in {(time.time()-t0)/60:.1f} min, R²={r2['overall']:.4f}")
            
            # Free memory
            del model, train_loader, val_loader, train_ds, val_ds, ds
            gc.collect()
        
        # Save spatial predictions for this seed
        np.savez(os.path.join(OUTPUT_DIR, f'spatial_pred_seed_{my_seed}.npz'),
                 pred_zero=spatial_preds['zero_fill'],
                 pred_replicate=spatial_preds['replicate_fill'],
                 pred_laplace=spatial_preds['laplace_fill'])
        
        all_my_results.append({'seed': my_seed, 'results': results})
        log(f"=== Seed {my_seed} complete ===")
    
    # =========================================================================
    # Gather and save
    # =========================================================================
    
    log(f"Finished all {len(my_seeds)} seeds, gathering results...")
    all_results = comm.gather(all_my_results, root=0)
    
    if rank == 0:
        total_time = (time.time() - start_time) / 60
        
        print("\n" + "=" * 60, flush=True)
        print(f"ALL COMPLETE in {total_time:.1f} minutes", flush=True)
        print("=" * 60, flush=True)
        
        # Flatten results from all ranks
        flat_results = []
        for rank_results in all_results:
            flat_results.extend(rank_results)
        
        # Aggregate
        stats = {m: {'overall': [], 'coastal': [], 'open_ocean': []} 
                 for m in ['zero_fill', 'replicate_fill', 'laplace_fill']}
        
        for item in flat_results:
            for method, r2 in item['results'].items():
                for region, val in r2.items():
                    stats[method][region].append(val)
        
        # Print summary
        print(f"\n{'Method':<18} {'Overall':>12} {'Coastal':>12} {'Open Ocean':>12}")
        print("-" * 56)
        for m in ['zero_fill', 'replicate_fill', 'laplace_fill']:
            o = np.mean(stats[m]['overall'])
            c = np.mean(stats[m]['coastal'])
            oo = np.mean(stats[m]['open_ocean'])
            o_std = np.std(stats[m]['overall'])
            c_std = np.std(stats[m]['coastal'])
            oo_std = np.std(stats[m]['open_ocean'])
            print(f"{m:<18} {o:.4f}±{o_std:.4f} {c:.4f}±{c_std:.4f} {oo:.4f}±{oo_std:.4f}")
        
        # Save JSON results
        with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
            json.dump({'seeds': SEEDS, 'results': flat_results, 'stats': stats}, 
                      f, indent=2, default=float)
        
        # Save target for plotting
        target_avg = np.nanmean(S_T, axis=0)  # Time-averaged target
        np.savez(os.path.join(OUTPUT_DIR, 'target_and_masks.npz'),
                 target=target_avg, ocean_mask=ocean_mask, coastal_mask=coastal_mask)
        
        # =====================================================================
        # SPATIAL MAPS (averaged over all seeds)
        # =====================================================================
        print("\nGenerating spatial maps...")
        
        # Load predictions from all seeds
        methods = ['zero_fill', 'replicate_fill', 'laplace_fill']
        method_labels = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
        
        all_preds = {m: [] for m in methods}
        for seed in SEEDS:
            data = np.load(os.path.join(OUTPUT_DIR, f'spatial_pred_seed_{seed}.npz'))
            all_preds['zero_fill'].append(data['pred_zero'])
            all_preds['replicate_fill'].append(data['pred_replicate'])
            all_preds['laplace_fill'].append(data['pred_laplace'])
        
        # Average and std across seeds
        avg_preds = {m: np.mean(all_preds[m], axis=0) for m in methods}
        std_preds = {m: np.std(all_preds[m], axis=0) for m in methods}
        avg_errors = {m: avg_preds[m] - target_avg for m in methods}
        
        # Coordinate setup
        ny, nx = target_avg.shape
        lat_edges = np.linspace(-90, 90, ny + 1)
        lon_edges = np.linspace(-180, 180, nx + 1)
        
        # =====================================================================
        # Plot 1: Target vs Predictions (2x2)
        # =====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        all_data = [target_avg] + [avg_preds[m] for m in methods]
        vmin, vmax = np.nanpercentile(np.concatenate([d.flatten() for d in all_data]), [2, 98])
        
        titles = ['True $S_T$ (Target)', 'Predicted $S_T$ (Zero-fill)', 
                  'Predicted $S_T$ (Replicate-fill)', 'Predicted $S_T$ (Laplace-fill)']
        data_list = [target_avg, avg_preds['zero_fill'], 
                     avg_preds['replicate_fill'], avg_preds['laplace_fill']]
        
        for ax, data, title in zip(axes.flat, data_list, titles):
            plot_data = np.ma.masked_invalid(np.flipud(data))
            im = ax.pcolormesh(lon_edges, lat_edges, plot_data,
                               cmap='viridis', vmin=vmin, vmax=vmax, shading='flat')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xticks(np.arange(-180, 181, 90))
            ax.set_yticks(np.arange(-90, 91, 45))
            ax.grid(True, alpha=0.3, linestyle='--')
        
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.6, pad=0.02)
        cbar.set_label('Subgrid Variance $S_T$ (K²)', fontsize=11)
        fig.suptitle(f'Subgrid Temperature Variance (n={len(SEEDS)} seeds average)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'ST_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ST_predictions.png")
        
        # =====================================================================
        # Plot 2: Errors (1x3)
        # =====================================================================
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        err_max = np.nanpercentile(np.abs(np.concatenate([avg_errors[m].flatten() for m in methods])), 98)
        
        for ax, method, label in zip(axes, methods, method_labels):
            plot_data = np.ma.masked_invalid(np.flipud(avg_errors[method]))
            im = ax.pcolormesh(lon_edges, lat_edges, plot_data,
                               cmap='RdBu_r', vmin=-err_max, vmax=err_max, shading='flat')
            ax.set_title(f'Error: {label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xticks(np.arange(-180, 181, 90))
            ax.set_yticks(np.arange(-90, 91, 45))
            ax.grid(True, alpha=0.3, linestyle='--')
        
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label('Error ($S_T^{pred} - S_T^{true}$) (K²)', fontsize=11)
        fig.suptitle(f'Prediction Errors (n={len(SEEDS)} seeds average)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'ST_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ST_errors.png")
        
        # =====================================================================
        # Plot 3: Zoomed Coastal Region (Gulf of Mexico)
        # =====================================================================
        lat_range = (15, 35)
        lon_range = (-100, -75)
        
        # Convert to indices
        lat_idx = (int((90 - lat_range[1]) / 180 * ny), int((90 - lat_range[0]) / 180 * ny))
        lon_idx = (int((lon_range[0] + 180) / 360 * nx), int((lon_range[1] + 180) / 360 * nx))
        
        lat_edges_zoom = np.linspace(lat_range[0], lat_range[1], lat_idx[1] - lat_idx[0] + 1)
        lon_edges_zoom = np.linspace(lon_range[0], lon_range[1], lon_idx[1] - lon_idx[0] + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Top: Predictions
        for ax, method, label in zip(axes[0], methods, method_labels):
            data = avg_preds[method][lat_idx[0]:lat_idx[1], lon_idx[0]:lon_idx[1]]
            plot_data = np.ma.masked_invalid(np.flipud(data))
            im1 = ax.pcolormesh(lon_edges_zoom, lat_edges_zoom, plot_data,
                                cmap='viridis', shading='auto')
            ax.set_title(f'Predicted $S_T$: {label}', fontsize=11)
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.grid(True, alpha=0.3)
        
        # Bottom: Errors
        for ax, method, label in zip(axes[1], methods, method_labels):
            data = avg_errors[method][lat_idx[0]:lat_idx[1], lon_idx[0]:lon_idx[1]]
            plot_data = np.ma.masked_invalid(np.flipud(data))
            im2 = ax.pcolormesh(lon_edges_zoom, lat_edges_zoom, plot_data,
                                cmap='RdBu_r', shading='auto')
            ax.set_title(f'Error: {label}', fontsize=11)
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.grid(True, alpha=0.3)
        
        fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8, label='$S_T$ (K²)')
        fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8, label='Error (K²)')
        fig.suptitle('Zoomed Coastal Region: Gulf of Mexico', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'ST_coastal_zoom.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ST_coastal_zoom.png")
        
        # =====================================================================
        # Plot 4: Error Metrics Bar Chart
        # =====================================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        metrics = {'RMSE': {}, 'MAE': {}}
        for method in methods:
            err = avg_errors[method]
            coastal_err = err[coastal_mask]
            ocean_err = err[ocean_mask & ~coastal_mask]
            
            metrics['RMSE'][method] = {'coastal': np.sqrt(np.nanmean(coastal_err**2)),
                                       'open_ocean': np.sqrt(np.nanmean(ocean_err**2))}
            metrics['MAE'][method] = {'coastal': np.nanmean(np.abs(coastal_err)),
                                      'open_ocean': np.nanmean(np.abs(ocean_err))}
        
        x = np.arange(3)
        width = 0.35
        colors = {'coastal': '#D55E00', 'open_ocean': '#009E73'}
        
        for ax, metric_name in zip(axes, ['RMSE', 'MAE']):
            coastal_vals = [metrics[metric_name][m]['coastal'] for m in methods]
            ocean_vals = [metrics[metric_name][m]['open_ocean'] for m in methods]
            
            ax.bar(x - width/2, coastal_vals, width, label='Coastal', color=colors['coastal'])
            ax.bar(x + width/2, ocean_vals, width, label='Open Ocean', color=colors['open_ocean'])
            
            ax.set_ylabel(f'{metric_name} (K²)')
            ax.set_title(f'{metric_name} by Region')
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels)
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        fig.suptitle(f'Spatial Error Metrics (n={len(SEEDS)} seeds)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'ST_error_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ST_error_metrics.png")
        
        # =====================================================================
        # Plot 5: R² Bar Chart
        # =====================================================================
        methods = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
        keys = ['zero_fill', 'replicate_fill', 'laplace_fill']
        colors = {'overall': '#0072B2', 'coastal': '#D55E00', 'open_ocean': '#009E73'}
        
        x = np.arange(3)
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for i, (region, label) in enumerate([('overall', 'Overall'), 
                                              ('coastal', 'Coastal'), 
                                              ('open_ocean', 'Open Ocean')]):
            means = [np.mean(stats[k][region]) for k in keys]
            stds = [np.std(stats[k][region]) for k in keys]
            ax.bar(x + (i-1)*0.25, means, 0.24, yerr=stds, capsize=4,
                   label=label, color=colors[region])
        
        ax.set_ylabel('R²')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim(0, 0.55)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'r2_comparison.png'), dpi=300)
        plt.close()
        
        print(f"\n{'='*60}")
        print(f"✓ All plots saved to {OUTPUT_DIR}/")
        print(f"  - ST_predictions.png")
        print(f"  - ST_errors.png")
        print(f"  - ST_coastal_zoom.png")
        print(f"  - ST_error_metrics.png")
        print(f"  - r2_comparison.png")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
