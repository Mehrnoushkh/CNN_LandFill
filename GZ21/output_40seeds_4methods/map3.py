import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# =============================================================================
# Load high-resolution land mask (for coastline contour)
# =============================================================================
with nc.Dataset('/scratch/10081/mkharghani/SST/updated_data/res005/data2023_01deg.nc') as ds:
    T_org   = np.array(ds['analysed_sst'][0, :, :])
    lat_org = np.array(ds['latitude'][:])
    lon_org = np.array(ds['longitude'][:])

land_mask_org = np.isnan(T_org) | (T_org == -999.0)
print("High-res land mask shape:", land_mask_org.shape)
print("Land fraction:", land_mask_org.mean())

# =============================================================================
# SETTINGS
# =============================================================================
OUTPUT_DIR = '.'
SEEDS = [ 5607, 5718, 5829, 5937, 6048, 6159, 6267, 6378, 6489, 6597,
    6708, 6819, 6927, 7038, 7149, 7257, 7368, 7479, 7587, 7698,
    7809, 7917, 8028, 8139, 8247, 8358, 8469, 8577, 8688, 8799,
    8907, 9018, 9129, 9237, 9348, 9459, 9567, 9678, 9789, 9897]

methods       = ['zero_fill', 'replicate_fill', 'laplace_fill']
method_labels = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']

# =============================================================================
# Load target, masks, and build coarse grid
# =============================================================================
masks        = np.load(os.path.join(OUTPUT_DIR, 'target_and_masks.npz'))
target_avg   = masks['target']
ocean_mask   = masks['ocean_mask']
coastal_mask = masks['coastal_mask']

ny, nx     = target_avg.shape
lat_coarse = np.linspace(-90, 90, ny)
lon_coarse = np.linspace(-180, 180, nx)

# =============================================================================
# Load and average spatial predictions across seeds
# =============================================================================
all_preds = {m: [] for m in methods}
for seed in SEEDS:
    fpath = os.path.join(OUTPUT_DIR, f'spatial_pred_seed_{seed}.npz')
    data  = np.load(fpath)
    all_preds['zero_fill'].append(data['pred_zero'])
    all_preds['replicate_fill'].append(data['pred_replicate'])
    all_preds['laplace_fill'].append(data['pred_laplace'])
    print(f"Loaded seed {seed}")

avg_preds  = {m: np.nanmean(all_preds[m], axis=0) for m in methods}
avg_errors = {m: avg_preds[m] - target_avg for m in methods}

print(f"Grid shape: {target_avg.shape}")
print(f"Lat range: {lat_coarse.min():.1f} to {lat_coarse.max():.1f}")
print(f"Lon range: {lon_coarse.min():.1f} to {lon_coarse.max():.1f}")

# =============================================================================
# Zoomed regional plots — predictions (top) and errors (bottom)
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

fig_width, fig_height = 9, 5  # tweak to taste

for region_name, region in regions.items():
    lat_min, lat_max = region['lat']
    lon_min, lon_max = region['lon']

    # --- Coarse grid zoom ---
    lat_idx = np.where((lat_coarse >= lat_min) & (lat_coarse <= lat_max))[0]
    lon_idx = np.where((lon_coarse >= lon_min) & (lon_coarse <= lon_max))[0]
    lat_zoom = lat_coarse[lat_idx]
    lon_zoom = lon_coarse[lon_idx]
    lon_z, lat_z = np.meshgrid(lon_zoom, lat_zoom)
    ocean_mask_zoom = ocean_mask[np.ix_(lat_idx, lon_idx)]

    # --- High-res grid zoom (for coastline contour) ---
    lat_idx_hr = np.where((lat_org >= lat_min) & (lat_org <= lat_max))[0]
    lon_idx_hr = np.where((lon_org >= lon_min) & (lon_org <= lon_max))[0]
    lat_hr = lat_org[lat_idx_hr]
    lon_hr = lon_org[lon_idx_hr]
    lon_hr2d, lat_hr2d = np.meshgrid(lon_hr, lat_hr)
    land_zoom_hr = land_mask_org[np.ix_(lat_idx_hr, lon_idx_hr)]

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)

    # --- Colormaps (NaN land → gray) ---
    cmap_pred = plt.cm.viridis.copy()
    #cmap_pred.set_bad('gray')
    cmap_err  = plt.cm.RdBu_r.copy()
    cmap_err.set_bad('gray')

    # --- Top row: predictions ---
    zoom_preds = []
    for m in methods:
        d = avg_preds[m][np.ix_(lat_idx, lon_idx)].copy()
       # d[~ocean_mask_zoom] = np.nan
        zoom_preds.append(d)

    vmin_z, vmax_z = np.nanpercentile(
        np.concatenate([d.flatten() for d in zoom_preds]), [2, 98])

    for col, (ax, data, label) in enumerate(zip(axes[0], zoom_preds, method_labels)):
        masked = np.ma.masked_invalid(data)
        im1 = ax.pcolormesh(lon_z, lat_z, masked,
                            cmap=cmap_pred, vmin=vmin_z, vmax=vmax_z,
                            shading='nearest', rasterized=True)
        ax.contour(lon_hr2d, lat_hr2d, land_zoom_hr.astype(float),
                   levels=[0.5], colors='black', linewidths=0.5)
        ax.set_title(f'({"abc"[col]}) {label}')
        if col == 0:
            ax.set_ylabel('Latitude ($^\\circ$)')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)

    # --- Bottom row: errors ---
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
                            shading='nearest',rasterized=True)
        ax.contour(lon_hr2d, lat_hr2d, land_zoom_hr.astype(float),
                   levels=[0.5], colors='black', linewidths=0.5)
        ax.set_title(f'({"def"[col]}) {label}')
        ax.set_xlabel('Longitude ($^\\circ$)')
        if col == 0:
            ax.set_ylabel('Latitude ($^\\circ$)')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)

    # --- Colorbars ---
    fig.subplots_adjust(right=0.87, hspace=0.15, wspace=0.1)

    cbar_ax1 = fig.add_axes([0.89, 0.53, 0.012, 0.35])
    cb1 = fig.colorbar(im1, cax=cbar_ax1)
    cb1.set_label('Predicted $S_T$ (K$^2$)', fontsize=7)
    cb1.ax.tick_params(labelsize=6)

    cbar_ax2 = fig.add_axes([0.89, 0.10, 0.012, 0.35])
    cb2 = fig.colorbar(im2, cax=cbar_ax2)
    cb2.set_label('Error (K$^2$)', fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    # --- Save ---
    plt.savefig(os.path.join(OUTPUT_DIR, f'{region["filename"]}_new.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{region["filename"]}_new.pdf'),
                bbox_inches='tight')
    plt.close()
    print(f"✓ {region['filename']}")
