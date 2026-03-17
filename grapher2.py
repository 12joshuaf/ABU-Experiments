import matplotlib.pyplot as plt
import numpy as np

models = ['Optimized ABU', 'Standard ABU', 'ReLU']
means  = [85.10, 84.91, 84.43]
stdevs = [1.48,  1.91,  2.05]

colors  = ['#4C72B0', '#DD8452', '#55A868']
markers = ['o', 's', '^']

fig, ax = plt.subplots(figsize=(8, 5))

for i, (model, mean, std) in enumerate(zip(models, means, stdevs)):
    ax.errorbar(
        i, mean,
        yerr=std,
        fmt=markers[i],
        color=colors[i],
        markersize=10,
        capsize=6,
        capthick=2,
        elinewidth=2,
        label=model
    )

ax.set_title('Testing Accuracy by Model', fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('Testing Accuracy (%)', fontsize=12)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(80, 90)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('testing_accuracy_by_model.png', dpi=150)
plt.show()