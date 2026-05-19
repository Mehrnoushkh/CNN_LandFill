import os
import json
import numpy as np
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
    'font.sans-serif': ['DejaVu Sans'],
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

gap = 10  # gap between region sections
section_width = n_seeds

for sec_i, (region, rlabel) in enumerate(zip(regions, region_labels)):
    x_offset = sec_i * (section_width + gap)

    for method, mlabel in zip(['laplace_fill', 'replicate_fill', 'zero_fill'],
                               ['Laplace-fill', 'Replicate-fill', 'Zero-fill']):
        values = np.array(r2_per_seed[(method, region)])

        cum_means = np.zeros((n_bootstrap, n_seeds))
        for b in range(n_bootstrap):
            order = np.random.permutation(n_seeds)
            shuffled = values[order]
            cum_means[b, :] = np.cumsum(shuffled) / np.arange(1, n_seeds + 1)

        x = np.arange(1, n_seeds + 1) + x_offset
        mean_curve = np.mean(cum_means, axis=0)
        q25 = np.percentile(cum_means, 25, axis=0)
        q75 = np.percentile(cum_means, 75, axis=0)

        ax.fill_between(x, q25, q75, alpha=0.15, color=colors[method])
        ax.plot(x, mean_curve, color=colors[method], linewidth=1.2)

    # Region label at center of section
    #center_x = x_offset + section_width / 2
    #ax.text(center_x, ax.get_ylim()[0] if sec_i > 0 else 0, rlabel,
    #        ha='center', va='top', fontsize=8, fontweight='bold')

# Fix the region labels after all data is plotted
y_bottom = ax.get_ylim()[0]
for sec_i, rlabel in enumerate(zip(region_labels)):
    pass  # handled below

# Vertical separators
for i in range(1, len(regions)):
    sep_x = i * (section_width + gap) - gap / 2
    ax.axvline(x=sep_x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# X-axis: region labels centered, remove numeric ticks
region_centers = [i * (section_width + gap) + section_width / 2 for i in range(len(regions))]
ax.set_xticks(region_centers)
ax.set_xticklabels(region_labels)

# Add seed-count ticks within each section
minor_ticks = []
minor_labels = []
for sec_i in range(len(regions)):
    x_offset = sec_i * (section_width + gap)
    for s in [1, 10, 20, 30, 40]:
        if s <= n_seeds:
            minor_ticks.append(s + x_offset)
            minor_labels.append(str(s))
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(minor_labels, minor=True, fontsize=5.5, color='gray')
ax.tick_params(axis='x', which='major', length=0, pad=15)
ax.tick_params(axis='x', which='minor', length=3, pad=2)

# Legend: Laplace, Replicate, Zero-fill (top to bottom)
legend_handles = [
    Line2D([0], [0], color=colors['laplace_fill'], linewidth=1.5, label='Laplace-fill'),
    Line2D([0], [0], color=colors['replicate_fill'], linewidth=1.5, label='Replicate-fill'),
    Line2D([0], [0], color=colors['zero_fill'], linewidth=1.5, label='Zero-fill'),
]
ax.legend(handles=legend_handles, framealpha=0.1, edgecolor='black', loc='lower right')

ax.set_ylabel('Cumulative mean $R^2$')
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seed_convergencev2.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'seed_convergencev2.tiff'), bbox_inches='tight', dpi=600)
plt.savefig(os.path.join(OUTPUT_DIR, 'seed_convergencev2.png'), bbox_inches='tight', dpi=600)
plt.close()
print(f"✓ seed_convergence ({n_seeds} seeds) — JAMES format")
