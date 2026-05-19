import os
import json
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = '.'
methods = ['zero_fill', 'replicate_fill', 'laplace_fill']
method_labels = ['Zero-fill', 'Replicate-fill', 'Laplace-fill']
regions = ['overall', 'coastal', 'open_ocean']
region_labels = ['Overall', 'Coastal', 'Open Ocean']

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'r') as f:
    data = json.load(f)

all_results = data['results']

r2_per_seed = {}
for m in methods:
    for r in regions:
        r2_per_seed[(m, r)] = [entry['results'][m][r] for entry in all_results]

n_seeds = len(all_results)
n_bootstrap = 200
colors = {'zero_fill': '#0072B2', 'replicate_fill': '#E69F00', 'laplace_fill': '#009E73'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, region, region_label in zip(axes, regions, region_labels):
    for method, mlabel in zip(methods, method_labels):
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
        q05 = np.percentile(cum_means, 5, axis=0)
        q95 = np.percentile(cum_means, 95, axis=0)

        ax.fill_between(x, q05, q95, alpha=0.1, color=colors[method])
        ax.fill_between(x, q25, q75, alpha=0.25, color=colors[method])
        ax.plot(x, mean_curve, color=colors[method], linewidth=2, label=mlabel)

    ax.set_title(region_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of seeds')
    ax.set_ylabel('Cumulative mean $R^2$')
    ax.set_xlim(1, n_seeds)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[2].legend(fontsize=9)
fig.suptitle('Convergence of $R^2$ with number of seeds\n(dark band = IQR, light band = 5th–95th percentile)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seed_convergence.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'seed_convergence.pdf'), bbox_inches='tight')
plt.close()
print(f"✓ seed_convergence ({n_seeds} seeds)")
