import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))

age = np.load(os.path.join(ROOT, "data", "age.npy"))[60000:].reshape(-1)
gender = np.load(os.path.join(ROOT, "data", "gender.npy"))[60000:].reshape(-1)
imgs = np.load(os.path.join(ROOT, "data", "test_images.npy"))

inds = np.where((gender == "male") & (age != -1) & (age >= 55) & (age < 100))[0]
valid = inds[inds < len(imgs)]

n = len(valid)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), dpi=150)
axes = np.array(axes).reshape(rows, cols)

for ax in axes.ravel():
    ax.axis("off")

for i, idx in enumerate(valid):
    ax = axes[i // cols, i % cols]
    ax.imshow(imgs[idx])
    ax.set_title(str(int(idx)), fontsize=6)
    ax.axis("off")

out = os.path.join(ROOT, "old_man_test_grid.png")
fig.tight_layout(pad=0.2)
fig.savefig(out, bbox_inches="tight")
print(out)
