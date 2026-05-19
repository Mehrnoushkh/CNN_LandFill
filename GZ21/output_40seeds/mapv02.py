"""
Standalone plotting script for multi-seed CNN spatial predictions.
Reads spatial_pred_seed_*.npz files and generates publication-quality plots.
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

mm_to_inch = 1 / 25.4
fig_width = 170 * mm_to_inch
fig_height = 70 * mm_to_inch

matplotlib.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 600,
})

# =============================================================================
# SETTINGS — adjust these to match your run
# =============================================================================

OUTPUT_DIR = '.'
#SEEDS = [42, 123, 456, 789, 1011, 12, 124, 458, 8, 65, 28,32,4567,222,924,576,184,30,90,967]
SEEDS = [42, 123, 456, 789, 1011, 12, 124, 458, 8, 65, 28,32,4567,222,924,576,184,30,90,967,
    # new 40 to add to the other 20
    137, 251, 303, 419, 538, 602, 717, 843, 955, 1089,
    1234, 1377, 1500, 1648, 1793, 1901, 2050, 2187, 2299, 2456,
    2601, 2777, 2888, 3010, 3159, 3333, 3478, 3599, 3721, 3888,
    4011, 4200, 4389, 4501, 4666, 4812, 4999, 5123, 5280, 5417,
]
PREPROCESSED_FILE = './../preprocessed_data4X.npz'

# Load target and masks
masks = np.load(os.path.join(OUTPUT_DIR, 'target_and_masks.npz'))
target_avg = masks['target']
ocean_mask = masks['ocean_mask']
coastal_mask = masks['coastal_mask']
ny, nx = target_avg.shape
lat_coarse = np.linspace(-90, 90, ny)
lon_coarse = np.linspace(-180, 180, nx)

# =============================================================================
# Load and average spatial predictions across seeds
# =============================================================================

methods = ['zero_fill', 'replicate_fill', 'laplace_fill']
method_labels = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']

all_preds = {m: [] for m in methods}

for seed in SEEDS:
    fpath = os.path.join(OUTPUT_DIR, f'spatial_pred_seed_{seed}.npz')
    data = np.load(fpath)
    all_preds['zero_fill'].append(data['pred_zero'])
    all_preds['replicate_fill'].append(data['pred_replicate'])
    all_preds['laplace_fill'].append(data['pred_laplace'])
    print(f"Loaded seed {seed}")

avg_preds = {m: np.nanmean(all_preds[m], axis=0) for m in methods}
avg_errors = {m: avg_preds[m] - target_avg for m in methods}

lon_2d, lat_2d = np.meshgrid(lon_coarse, lat_coarse)

print(f"Grid shape: {target_avg.shape}")
print(f"Lat range: {lat_coarse.min():.1f} to {lat_coarse.max():.1f}")
print(f"Lon range: {lon_coarse.min():.1f} to {lon_coarse.max():.1f}")

# =============================================================================
# Plot 0: Coarse-grained temperature T_bar (from preprocessed data)
# =============================================================================
if os.path.exists(PREPROCESSED_FILE):
    print("\nLoading preprocessed data for T_bar plot...")
    preproc = np.load(PREPROCESSED_FILE)
    T_coarse_zero = preproc['T_coarse_zero']
    T_coarse_rep = preproc['T_coarse_replicate']
    T_coarse_lap = preproc['T_coarse_laplace']

    # Time-average
    T_avg_zero = np.mean(T_coarse_zero, axis=0)
    T_avg_rep = np.mean(T_coarse_rep, axis=0)
    T_avg_lap = np.mean(T_coarse_lap, axis=0)

    # Original = same as any method over ocean, NaN over land
    T_avg_orig = T_avg_zero.copy()
    T_avg_orig[~ocean_mask] = np.nan
    MIN =np.nanmin( T_avg_orig)
    MAX = np.nanmax(T_avg_orig)
    
    del T_coarse_zero, T_coarse_rep, T_coarse_lap  # free memory

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    titles = [r'$\overline{T}$ (Original, ocean only)',
              r'$\overline{T}$ (Zero-fill)',
              r'$\overline{T}$ (Replicate-fill)',
              r'$\overline{T}$ (Laplace-fill)']
    data_list = [T_avg_orig, T_avg_zero, T_avg_rep, T_avg_lap]

    vmin, vmax = np.nanpercentile(T_avg_orig, [2, 100])

    for ax, plot_data, title in zip(axes.flat, data_list, titles):
        plot_data[~ocean_mask] = np.nan
        masked = np.ma.masked_invalid(plot_data)
        im = ax.pcolormesh(lon_2d, lat_2d, masked,
                           cmap='RdYlBu_r', vmin=272, vmax=MAX)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Temperature (K)', fontsize=11)

    fig.suptitle('coarsened temperature $\\overline{T}$ (average over one year)',
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, 'T_bar_comparison.png'), dpi=450, bbox_inches='tight')

    #plt.savefig(os.path.join(OUTPUT_DIR, 'T_bar_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ T_bar_comparison")
else:
    print(f"⚠ No {PREPROCESSED_FILE} found, skipping T_bar plot")

# =============================================================================
# Plot 1: Target vs Predictions (2x2)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 9))

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
    #masked = np.ma.masked_invalid(plot_data)
    im = ax.pcolormesh(lon_2d, lat_2d, masked,
                       cmap='viridis', vmin=vmin, vmax=vmax, shading='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(lon_coarse.min(), lon_coarse.max())
    ax.set_ylim(lat_coarse.min(), lat_coarse.max())
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, alpha=0.3, linestyle='--')

fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Subgrid Variance $S_T$ (K²)', fontsize=11)

fig.suptitle(f'Subgrid Temperature Variance (n={len(SEEDS)} seeds average)',
             fontsize=14, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, 'ST_predictions.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'ST_predictions.pdf'), bbox_inches='tight')
plt.close()
print("✓ ST_predictions")

# =============================================================================
# Plot 2: Errors (1x3)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

err_max = np.nanpercentile(np.abs(np.concatenate([avg_errors[m].flatten() for m in methods])), 98)

for ax, method, label in zip(axes, methods, method_labels):
    err_masked = avg_errors[method].copy()
    err_masked[~ocean_mask] = np.nan
    masked = np.ma.masked_invalid(err_masked)
    #masked = np.ma.masked_invalid(avg_errors[method])
    im = ax.pcolormesh(lon_2d, lat_2d, masked,
                       cmap='RdBu_r', vmin=-err_max, vmax=err_max, shading='nearest')
    ax.set_title(f'Error: {label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_xlim(lon_coarse.min(), lon_coarse.max())
    ax.set_ylim(lat_coarse.min(), lat_coarse.max())
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, alpha=0.3, linestyle='--')

fig.subplots_adjust(right=0.88, wspace=0.25)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Error ($S_T^{pred} - S_T^{true}$) (K²)', fontsize=11)

fig.suptitle(f'Prediction Errors (n={len(SEEDS)} seeds average)', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, 'ST_errors.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'ST_errors.pdf'), bbox_inches='tight')
plt.close()
print("✓ ST_errors")
#__________


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Shared color limits
err_max = np.nanpercentile(np.abs(np.concatenate([avg_errors[m].flatten() for m in methods])), 98)
abs_max = np.nanpercentile(np.concatenate([np.abs(avg_errors[m]).flatten() for m in methods]), 98)

for col, (method, label) in enumerate(zip(methods, method_labels)):
    err = avg_errors[method].copy()
    err[~ocean_mask] = np.nan
    abs_err = np.abs(err)
    masked_err = np.ma.masked_invalid(err)
    masked_abs = np.ma.masked_invalid(abs_err)

    # Top row: signed error (diverging)
    ax_top = axes[0, col]
    im_top = ax_top.pcolormesh(lon_2d, lat_2d, masked_err,
                               cmap='RdBu_r', vmin=-err_max, vmax=err_max, shading='nearest')
    ax_top.set_title(f'Error: {label}', fontsize=12, fontweight='bold')
    ax_top.set_xlabel('Longitude (°)')
    ax_top.set_ylabel('Latitude (°)')
    ax_top.set_xlim(lon_coarse.min(), lon_coarse.max())
    ax_top.set_ylim(lat_coarse.min(), lat_coarse.max())
    ax_top.set_xticks(np.arange(-180, 181, 60))
    ax_top.set_yticks(np.arange(-90, 91, 30))
    ax_top.grid(True, alpha=0.3, linestyle='--')

    # Bottom row: absolute error (sequential)
    ax_bot = axes[1, col]
    im_bot = ax_bot.pcolormesh(lon_2d, lat_2d, masked_abs,
                               cmap='Reds', vmin=0, vmax=abs_max, shading='nearest')
    ax_bot.set_title(f'|Error|: {label}', fontsize=12, fontweight='bold')
    ax_bot.set_xlabel('Longitude (°)')
    ax_bot.set_ylabel('Latitude (°)')
    ax_bot.set_xlim(lon_coarse.min(), lon_coarse.max())
    ax_bot.set_ylim(lat_coarse.min(), lat_coarse.max())
    ax_bot.set_xticks(np.arange(-180, 181, 60))
    ax_bot.set_yticks(np.arange(-90, 91, 30))
    ax_bot.grid(True, alpha=0.3, linestyle='--')

# Colorbars
fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.30)
cbar_ax1 = fig.add_axes([0.90, 0.53, 0.015, 0.35])
cbar1 = fig.colorbar(im_top, cax=cbar_ax1)
cbar1.set_label('Error ($S_T^{pred} - S_T^{true}$) (K²)', fontsize=11)

cbar_ax2 = fig.add_axes([0.90, 0.10, 0.015, 0.35])
cbar2 = fig.colorbar(im_bot, cax=cbar_ax2)
cbar2.set_label('|Error| (K²)', fontsize=11)

fig.suptitle(f'Prediction Errors (n={len(SEEDS)} seeds average)', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, 'ST_errors_update.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ ST_errors22")

# =============================================================================
# Plot 3: Zoomed Coastal Region (Gulf of Mexico)
# =============================================================================
#lat_min, lat_max = 15, 35
#lon_min, lon_max = -100, -75
lat_min, lat_max = 20, 55
lon_min, lon_max = -60, 0

lat_idx = np.where((lat_coarse >= lat_min) & (lat_coarse <= lat_max))[0]
lon_idx = np.where((lon_coarse >= lon_min) & (lon_coarse <= lon_max))[0]
lat_zoom = lat_coarse[lat_idx]
lon_zoom = lon_coarse[lon_idx]
lon_z, lat_z = np.meshgrid(lon_zoom, lat_zoom)
ocean_mask_zoom = ocean_mask[np.ix_(lat_idx, lon_idx)]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Top row: Predictions
zoom_preds = [avg_preds[m][np.ix_(lat_idx, lon_idx)] for m in methods]
vmin_z, vmax_z = np.nanpercentile(np.concatenate([d.flatten() for d in zoom_preds]), [2, 98])

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
  #  ax.set_xlabel('Longitude (°)')
  #  ax.set_ylabel('Latitude (°)')
    ax.grid(True, alpha=0.3)

fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.3)
cbar_ax1 = fig.add_axes([0.90, 0.53, 0.015, 0.35])
fig.colorbar(im1, cax=cbar_ax1, label='$S_T$ (K²)')
cbar_ax2 = fig.add_axes([0.90, 0.10, 0.015, 0.35])
fig.colorbar(im2, cax=cbar_ax2, label='Error (K²)')

fig.suptitle(f'Zoomed Coastal Region: Gulf stream (n={len(SEEDS)} seeds avg)',
             fontsize=13, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, 'ST_coastal_gulf_stream.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'ST_coastal_gulf_stream.pdf'), bbox_inches='tight')
plt.close()
print("✓ ST_coastal_zoom")
# =============================================================================
# Plot 3: Zoomed Coastal Regions (Gulf Stream, Kuroshio, Drake Passage)
# =============================================================================
regions = {
    'Gulf Stream': {
        'lat': (20, 55), 'lon': (-80, -10),
        'filename': 'ST_coastal_gulf_stream'
    },
    'Kuroshio Current': {
        'lat': (20, 50), 'lon': (120, 180),
        'filename': 'ST_coastal_kuroshio'
    },
    'Drake Passage': {
        'lat': (-70, -50), 'lon': (-80, -50),
        'filename': 'ST_coastal_drake'
    },
}

for region_name, region in regions.items():
    lat_min, lat_max = region['lat']
    lon_min, lon_max = region['lon']

    lat_idx = np.where((lat_coarse >= lat_min) & (lat_coarse <= lat_max))[0]
    lon_idx = np.where((lon_coarse >= lon_min) & (lon_coarse <= lon_max))[0]
    lat_zoom = lat_coarse[lat_idx]
    lon_zoom = lon_coarse[lon_idx]
    lon_z, lat_z = np.meshgrid(lon_zoom, lat_zoom)
    ocean_mask_zoom = ocean_mask[np.ix_(lat_idx, lon_idx)]

    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height),rasterized = True, sharex = True, sharey=True)

    # Top row: Predictions
    zoom_preds = []
    for m in methods:
        d = avg_preds[m][np.ix_(lat_idx, lon_idx)].copy()
        d[~ocean_mask_zoom] = np.nan
        zoom_preds.append(d)
    vmin_z, vmax_z = np.nanpercentile(
        np.concatenate([d.flatten() for d in zoom_preds]), [2, 98])
    cmap_pred = plt.cm.viridis.copy()
    cmap_pred.set_bad('black')

    cmap_err = plt.cm.RdBu_r.copy()
    cmap_err.set_bad('black')
    for ax, data, label in zip(axes[0], zoom_preds, method_labels):
        masked = np.ma.masked_invalid(data)
        im1 = ax.pcolormesh(lon_z, lat_z, masked,
                            cmap=cmap_pred, vmin=vmin_z, vmax=vmax_z, shading='nearest')
        ax.set_title(f'Predicted $S_T$: {label}', fontsize=11)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3)

    # Bottom row: Errors
    zoom_errors = []
    for m in methods:
        d = avg_errors[m][np.ix_(lat_idx, lon_idx)].copy()
        d[~ocean_mask_zoom] = np.nan
        zoom_errors.append(d)
    err_max_z = np.nanpercentile(
        np.abs(np.concatenate([d.flatten() for d in zoom_errors])), 98)

    for ax, data, label in zip(axes[1], zoom_errors, method_labels):
        masked = np.ma.masked_invalid(data)
        im2 = ax.pcolormesh(lon_z, lat_z, masked,
                            cmap=cmap_err, vmin=-err_max_z, vmax=err_max_z, shading='nearest')
        ax.set_title(f'Error: {label}', fontsize=11)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(right=0.88, hspace=0.1, wspace=0.3)
    cbar_ax1 = fig.add_axes([0.90, 0.53, 0.015, 0.35])
    fig.colorbar(im1, cax=cbar_ax1, label='$S_T$ (K²)')
    cbar_ax2 = fig.add_axes([0.90, 0.10, 0.015, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label='Error (K²)')

    fig.suptitle(f'Zoomed Region: {region_name} (n={len(SEEDS)} seeds avg)',
                 fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{region["filename"]}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{region["filename"]}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ {region['filename']}")



import matplotlib
mm_to_inch = 1 / 25.4
fig_width = 170 * mm_to_inch
fig_height = 100 * mm_to_inch  # taller for 2 rows

matplotlib.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 600,
})

regions = {
    'Gulf Stream': {
        'lat': (20, 55), 'lon': (-80, -10),
        'filename': 'ST_coastal_gulf_stream'
    },
    'Kuroshio Current': {
        'lat': (20, 50), 'lon': (120, 180),
        'filename': 'ST_coastal_kuroshio'
    },
    'Drake Passage': {
        'lat': (-70, -50), 'lon': (-80, -50),
        'filename': 'ST_coastal_drake'
    },
}

for region_name, region in regions.items():
    lat_min, lat_max = region['lat']
    lon_min, lon_max = region['lon']

    lat_idx = np.where((lat_coarse >= lat_min) & (lat_coarse <= lat_max))[0]
    lon_idx = np.where((lon_coarse >= lon_min) & (lon_coarse <= lon_max))[0]
    lat_zoom = lat_coarse[lat_idx]
    lon_zoom = lon_coarse[lon_idx]
    lon_z, lat_z = np.meshgrid(lon_zoom, lat_zoom)
    ocean_mask_zoom = ocean_mask[np.ix_(lat_idx, lon_idx)]

    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)

    # Top row: Predictions
    zoom_preds = []
    for m in methods:
        d = avg_preds[m][np.ix_(lat_idx, lon_idx)].copy()
        d[~ocean_mask_zoom] = np.nan
        zoom_preds.append(d)
    vmin_z, vmax_z = np.nanpercentile(
        np.concatenate([d.flatten() for d in zoom_preds]), [2, 98])
    cmap_pred = plt.cm.viridis.copy()
    cmap_pred.set_bad('black')

    cmap_err = plt.cm.RdBu_r.copy()
    cmap_err.set_bad('black')

    for col, (ax, data, label) in enumerate(zip(axes[0], zoom_preds, method_labels)):
        masked = np.ma.masked_invalid(data)
        im1 = ax.pcolormesh(lon_z, lat_z, masked,
                            cmap=cmap_pred, vmin=vmin_z, vmax=vmax_z,
                            shading='nearest', rasterized=True)
        ax.set_title(f'({"abc"[col]}) {label}')
        if col == 0:
            ax.set_ylabel('Latitude ($^\\circ$)')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)

    # Bottom row: Errors
    zoom_errors = []
    for m in methods:
        d = avg_errors[m][np.ix_(lat_idx, lon_idx)].copy()
        d[~ocean_mask_zoom] = np.nan
        zoom_errors.append(d)
    err_max_z = np.nanpercentile(
        np.abs(np.concatenate([d.flatten() for d in zoom_errors])), 98)

    for col, (ax, data, label) in enumerate(zip(axes[1], zoom_errors, method_labels)):
        masked = np.ma.masked_invalid(data)
        im2 = ax.pcolormesh(lon_z, lat_z, masked,
                            cmap=cmap_err, vmin=-err_max_z, vmax=err_max_z,
                            shading='nearest', rasterized=True)
        ax.set_title(f'({"def"[col]}) {label}')
        ax.set_xlabel('Longitude ($^\\circ$)')
        if col == 0:
            ax.set_ylabel('Latitude ($^\\circ$)')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)

    # Colorbars
    fig.subplots_adjust(right=0.87, hspace=0.15, wspace=0.1)
    cbar_ax1 = fig.add_axes([0.89, 0.53, 0.012, 0.35])
    cb1 = fig.colorbar(im1, cax=cbar_ax1)
    cb1.set_label('Predicted $S_T$ (K$^2$)', fontsize=7)
    cb1.ax.tick_params(labelsize=6)

    cbar_ax2 = fig.add_axes([0.89, 0.10, 0.012, 0.35])
    cb2 = fig.colorbar(im2, cax=cbar_ax2)
    cb2.set_label('Error (K$^2$)', fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    # Row labels on the left margin
    #fig.text(0.01, 0.72, 'Predicted $S_T$', va='center', ha='left',
    #         fontsize=8, fontweight='bold', rotation=90)
    #fig.text(0.01, 0.30, 'Error', va='center', ha='left',
    #         fontsize=8, fontweight='bold', rotation=90)

    plt.savefig(os.path.join(OUTPUT_DIR, f'{region["filename"]}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{region["filename"]}.tiff'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{region["filename"]}.png'), bbox_inches='tight', dpi=600)
    plt.close()
    print(f"✓ {region['filename']}")



# =============================================================================
# Plot 4: R² Bar Chart (load from results.json)
# =============================================================================
COLORS = ['#3498DB', '#E67E22', '#2ECC71']

results_file = os.path.join(OUTPUT_DIR, 'results.json')
if os.path.exists(results_file):
    with open(results_file) as f:
        raw = json.load(f)

    # Extract per-seed R² values
    stats = {m: {'overall': [], 'coastal': [], 'open_ocean': []}
             for m in methods}
    for item in raw['results']:
        for method, r2 in item['results'].items():
            for region, val in r2.items():
                stats[method][region].append(val)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(method_labels))
    width = 0.25
    regions = ['overall', 'coastal', 'open_ocean']
    region_labels = ['Overall', 'Coastal', 'Open Ocean']

    for i, (region, label, color) in enumerate(zip(regions, region_labels, COLORS)):
        means = [np.mean(stats[m][region]) for m in methods]
        stds = [np.std(stats[m][region]) for m in methods]

        bars = ax.bar(x + (i - 1) * width, means, width,
                      yerr=stds, capsize=4,
                      label=label, color=color,
                      edgecolor='white', linewidth=0.7,
                      error_kw={'linewidth': 1.2, 'capthick': 1.2})

        for j, (bar, mean_val) in enumerate(zip(bars, means)):
            ax.annotate(f'{mean_val:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + stds[j] + 0.005),
                        ha='center', va='bottom', fontsize=8, fontweight='medium')

    ax.set_ylabel('$R^2$ Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=11)
    ax.set_ylim(0, max([np.mean(stats[m]['open_ocean']) for m in methods]) * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods_errorbars.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'r2_three_methods_errorbars.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ r2_three_methods_errorbars")
else:
    print(f"⚠ No {results_file} found, skipping R² plot")

# =============================================================================
# Plot 5: RMSE/MAE bar chart
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

metric_data = {'RMSE': {}, 'MAE': {}}
for method in methods:
    err = avg_errors[method]
    coastal_err = err[coastal_mask]
    open_err = err[ocean_mask & ~coastal_mask]

    metric_data['RMSE'][method] = {
        'coastal': np.sqrt(np.nanmean(coastal_err**2)),
        'open_ocean': np.sqrt(np.nanmean(open_err**2))
    }
    metric_data['MAE'][method] = {
        'coastal': np.nanmean(np.abs(coastal_err)),
        'open_ocean': np.nanmean(np.abs(open_err))
    }

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
print("✓ ST_error_metrics")
# Plot 5: RMSE / NRMSE / MAE bar chart
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = {'RMSE': {}, 'NRMSE': {}, 'MAE': {}}

# Region-specific RMS of true S_T for normalization
coastal_target = target_avg[coastal_mask]
ocean_target = target_avg[ocean_mask & ~coastal_mask]

rms_coastal = np.sqrt(np.nanmean(coastal_target**2))
rms_ocean = np.sqrt(np.nanmean(ocean_target**2))

for method in methods:
    err = avg_errors[method]
    coastal_err = err[coastal_mask]
    ocean_err = err[ocean_mask & ~coastal_mask]

    rmse_c = np.sqrt(np.nanmean(coastal_err**2))
    rmse_o = np.sqrt(np.nanmean(ocean_err**2))

    metrics['RMSE'][method] = {'coastal': rmse_c, 'open_ocean': rmse_o}
    metrics['NRMSE'][method] = {'coastal': rmse_c / rms_coastal, 'open_ocean': rmse_o / rms_ocean}
    metrics['MAE'][method] = {'coastal': np.nanmean(np.abs(coastal_err)),
                              'open_ocean': np.nanmean(np.abs(ocean_err))}

x = np.arange(3)
width = 0.35
colors = {'coastal': '#D55E00', 'open_ocean': '#009E73'}

for ax, metric_name in zip(axes, ['RMSE', 'NRMSE', 'MAE']):
    coastal_vals = [metrics[metric_name][m]['coastal'] for m in methods]
    ocean_vals = [metrics[metric_name][m]['open_ocean'] for m in methods]
    ax.bar(x - width/2, coastal_vals, width, label='Coastal', color=colors['coastal'])
    ax.bar(x + width/2, ocean_vals, width, label='Open Ocean', color=colors['open_ocean'])

    if metric_name == 'NRMSE':
        ax.set_ylabel('NRMSE (unitless)')
    else:
        ax.set_ylabel(f'{metric_name} (K²)')

    ax.set_title(f'{metric_name} by Region')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(f'Normalized Spatial Error Metrics (n={len(SEEDS)} seeds avg)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ST_error_metricsi_normalize.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: ST_error_metrics")
print(f"\nAll plots saved to {OUTPUT_DIR}/")
