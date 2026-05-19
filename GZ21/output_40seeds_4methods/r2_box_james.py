import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# JAMES two-column figure: 170mm wide, height ≤ 228mm
mm_to_inch = 1 / 25.4
fig_width = 170 * mm_to_inch   # 6.69 inches
fig_height = 80 * mm_to_inch   # ~3.15 inches (compact for single row)

# Set font sizes to match 8pt at final print size
matplotlib.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42,       # TrueType fonts in PDF (required by many journals)
    'ps.fonttype': 42,
    'savefig.dpi': 600,        # line art resolution
})

OUTPUT_DIR = '.'
methods = ['zero_fill', 'replicate_fill', 'laplace_fill']
method_labels = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
regions = ['overall', 'coastal', 'open_ocean']
region_labels = ['Overall', 'Coastal', 'Open Ocean']
colors = {'zero_fill': '#0072B2', 'replicate_fill': '#E69F00', 'laplace_fill': '#009E73'}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'r') as f:
    data = json.load(f)
all_results = data['results']

r2_per_seed = {}
for m in methods:
    for r in regions:
        r2_per_seed[(m, r)] = [entry['results'][m][r] for entry in all_results]

n_seeds = len(all_results)
n_regions = len(regions)
n_methods = len(methods)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

spacing = 2.4
width = 0.6
offsets = np.array([-width, 0, width])

positions_all = []
data_all = []
color_all = []

for i, (region, rlabel) in enumerate(zip(regions, region_labels)):
    center = i * spacing
    for j, (method, mlabel) in enumerate(zip(methods, method_labels)):
        pos = center + offsets[j]
        positions_all.append(pos)
        data_all.append(r2_per_seed[(method, region)])
        color_all.append(colors[method])

bp = ax.boxplot(data_all, positions=positions_all, widths=width * 0.8,
                patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white',
                               markeredgecolor='black', markersize=3),
                medianprops=dict(color='black', linewidth=1),
                flierprops=dict(marker='o', markersize=2.5, alpha=0.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                boxprops=dict(linewidth=0.8))

for patch, color in zip(bp['boxes'], color_all):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Individual seed points (jittered)
for idx, (pos, color) in enumerate(zip(positions_all, color_all)):
    values = data_all[idx]
    jitter = np.random.normal(0, 0.05, size=len(values))
    ax.scatter(np.full(len(values), pos) + jitter, values,
               color=color, alpha=0.3, s=5, zorder=0)

# Vertical separators between regions
group_centers = [i * spacing for i in range(n_regions)]
for i in range(n_regions - 1):
    midpoint = (group_centers[i] + group_centers[i + 1]) / 2
    ax.axvline(x=midpoint, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# X-axis
ax.set_xticks(group_centers)
ax.set_xticklabels(region_labels)

# Legend
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], alpha=0.7) for m in methods]
ax.legend(handles, method_labels, loc='lower right', framealpha=0.9, edgecolor='gray')

ax.set_ylabel('$R^2$')
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'R2_boxplot.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'R2_boxplot.tiff'), bbox_inches='tight', dpi=600)
plt.savefig(os.path.join(OUTPUT_DIR, 'R2_boxplot.png'), bbox_inches='tight', dpi=600)
plt.close()
print(f"✓ R2_boxplot ({n_seeds} seeds) — JAMES format")
