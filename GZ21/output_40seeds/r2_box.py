import os
import json
import numpy as np
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(figsize=(10, 6))

n_regions = len(regions)
n_methods = len(methods)
width = 0.6
spacing = 2.5  # space between region groups
offsets = np.array([-width, 0, width])  # offset for each method within a group

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
                               markeredgecolor='black', markersize=5),
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=4, alpha=0.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2))

for patch, color in zip(bp['boxes'], color_all):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Individual seed points (jittered)
for idx, (pos, color) in enumerate(zip(positions_all, color_all)):
    values = data_all[idx]
    jitter = np.random.normal(0, 0.06, size=len(values))
    ax.scatter(np.full(len(values), pos) + jitter, values,
               color=color, alpha=0.2, s=10, zorder=0)

# X-axis: region labels at group centers
group_centers = [i * spacing for i in range(n_regions)]
ax.set_xticks(group_centers)
ax.set_xticklabels(region_labels, fontsize=12)

# Legend
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], alpha=0.7) for m in methods]
ax.legend(handles, method_labels, fontsize=11, loc='lower right')

ax.set_ylabel('$R^2$', fontsize=13)
ax.set_title(f'$R^2$ distribution across {n_seeds} seeds', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i in range(n_regions - 1):
    midpoint = (group_centers[i] + group_centers[i + 1]) / 2
    ax.axvline(x=midpoint, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'R2_boxplot.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'R2_boxplot.pdf'), bbox_inches='tight')
plt.close()
print(f"✓ R2_boxplot ({n_seeds} seeds)")
