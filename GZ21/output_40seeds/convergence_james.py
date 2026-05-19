import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

mm_to_inch = 1 / 25.4
fig_width = 85 * mm_to_inch    # single-column width
fig_height = 85 * mm_to_inch

matplotlib.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 600,
})

OUTPUT_DIR = '.'
methods = ['zero_fill', 'replicate_fill', 'laplace_fill']
method_labels = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
regions = ['overall', 'coastal', 'open_ocean']
region_labels = ['Overall', 'Coastal', 'Open Ocean']
colors = {'zero_fill': '#0072B2', 'replicate_fill': '#E69F00', 'laplace_fill': '#009E73'}
linestyles = {'overall': '-', 'coastal': '--', 'open_ocean': ':'}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'r') as f:
    data = json.load(f)
all_results = data['results']

r2_per_seed = {}
for m in methods:
    for r in regions:
        r2_per_seed[(m, r)] = [entry['results'][m][r] for entry in all_results]

n_seeds = len(all_results)
n_bootstrap = 200

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Plot order: Laplace, Replicate, Zero-fill (top to bottom in legend)
method_order = ['laplace_fill', 'replicate_fill', 'zero_fill']
label_order = ['Laplace-fill', 'Replicate-fill', 'Zero-fill']

for method, mlabel in zip(method_order, label_order):
    for region, rlabel, ls in zip(regions, region_labels, ['-', '--', ':']):
        values = np.array(r2_per_seed[(method, region)])

        cum_means = np.zeros((n_bootstrap, n_seeds))
        for b in range(n_bootstrap):
            order = np.random.permutation(n_seeds)
            shuffled = values[order]
            cum_means[b, :] = np.cumsum(shuffled) / np.arange(1, n_seeds + 1)

        x = np.arange(1, n_seeds + 1)
        mean_curve = np.mean(cum_means, axis=0)
        q25 = np.percentile(cum_means, 25, axis=0)
        q75 = np.percentile(cum_means, 75, axis=0)

        ax.fill_between(x, q25, q75, alpha=0.12, color=colors[method])
        ax.plot(x, mean_curve, color=colors[method], linewidth=1.2,
                linestyle=ls, label=f'{mlabel} — {rlabel}')

ax.set_xlabel('Number of seeds')
ax.set_ylabel('Cumulative mean $R^2$')
ax.set_xlim(1, n_seeds)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Custom legend: methods by color, regions by line style
from matplotlib.lines import Line2D
legend_handles = []
# Method entries (color)
for method, mlabel in zip(method_order, label_order):
    legend_handles.append(Line2D([0], [0], color=colors[method], linewidth=1.5, label=mlabel))
# Separator
legend_handles.append(Line2D([0], [0], color='none', label=''))
# Region entries (line style)
for region, rlabel, ls in zip(regions, region_labels, ['-', '--', ':']):
    legend_handles.append(Line2D([0], [0], color='gray', linewidth=1.2,
                                 linestyle=ls, label=rlabel))

ax.legend(handles=legend_handles, framealpha=0.9, edgecolor='gray', loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seed_convergence.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'seed_convergence.tiff'), bbox_inches='tight', dpi=600)
plt.savefig(os.path.join(OUTPUT_DIR, 'seed_convergence.png'), bbox_inches='tight', dpi=600)
plt.close()
print(f"✓ seed_convergence ({n_seeds} seeds) — JAMES format")
