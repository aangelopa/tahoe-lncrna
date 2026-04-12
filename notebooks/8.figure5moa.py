"""

Points with 'unclear' MOA are shown in gray background.
Points with known MOA are colored and labeled.

Output:
  figure5_de_moa.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# CONFIGURATION

EMBED_DIR = os.path.expanduser("~/Thesis/Data/dim_reduction/")
DRUG_META = os.path.expanduser("~/Thesis/Data/drug_metadata.parquet")
OUTPUT_DIR = EMBED_DIR
MOA_COL = "moa-fine"


# 1. LOAD DATA

print("1. Loading data...")

tsne = np.load(os.path.join(EMBED_DIR, "tsne_embedding.npy"))
nnmds = np.load(os.path.join(EMBED_DIR, "nnmds_embedding.npy"))
metadata = pd.read_parquet(os.path.join(EMBED_DIR, "metadata.parquet"))
drug_meta = pd.read_parquet(DRUG_META)

print(f"   Embeddings: {tsne.shape[0]} points")
print(f"   Drug metadata: {drug_meta.shape[0]} drugs")


# 2. MERGE MOA INTO METADATA

print("\n2. Merging MOA annotations...")

# Strip whitespace from drug names
metadata["drug"] = metadata["drug"].str.strip()
drug_meta["drug"] = drug_meta["drug"].str.strip()

# Merge
metadata = metadata.merge(drug_meta[["drug", MOA_COL]], on="drug", how="left")

# Stats
n_total = len(metadata)
n_with_moa = metadata[MOA_COL].notna().sum()
n_unclear = (metadata[MOA_COL] == "unclear").sum()
n_known = n_with_moa - n_unclear
print(f"   Total comparisons: {n_total}")
print(f"   With MOA annotation: {n_with_moa}")
print(f"   Known MOA: {n_known}")
print(f"   Unclear MOA: {n_unclear}")
print(f"   No annotation: {n_total - n_with_moa}")

# MOA value counts (excluding unclear)
moa_counts = metadata[metadata[MOA_COL] != "unclear"][MOA_COL].value_counts()
print(f"\n   MOA categories (excluding unclear):")
for moa, count in moa_counts.items():
    print(f"     {moa}: {count}")


# 3. PLOT FIGURE 5D/E

print("\n3. Generating plots...")

moa_values = metadata[MOA_COL].values

# Define which points are "known MOA" vs background
is_known = pd.notna(moa_values) & (moa_values != "unclear")
is_background = ~is_known

# Color map for known MOAs
unique_moas = sorted(set(moa_values[is_known]))
n_moas = len(unique_moas)


cmap_colors = plt.cm.tab20(np.linspace(0, 1, 20))
if n_moas > 20:
    cmap_colors = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
    ])

moa_color = {moa: cmap_colors[i % len(cmap_colors)] for i, moa in enumerate(unique_moas)}

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, embedding, title in [
    (axes[0], tsne, "t-SNE"),
    (axes[1], nnmds, "NNMDS"),
]:
    # Background: unclear + no MOA in gray
    ax.scatter(
        embedding[is_background, 0], embedding[is_background, 1],
        c="lightgray", s=1, alpha=0.15, rasterized=True, zorder=1, label="_nolegend_"
    )

    # Plot each MOA category separately for legend
    for moa in unique_moas:
        mask = moa_values == moa
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[moa_color[moa]], s=3, alpha=0.6, rasterized=True,
            zorder=2, label=moa
        )

    ax.set_title(f"{title} — Drug MOA", fontsize=13)
    ax.set_xlabel(f"{title} 1")
    ax.set_ylabel(f"{title} 2")

# Single legend on the right
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(
    handles, labels, title="Mechanism of Action",
    bbox_to_anchor=(1.05, 1), loc="upper left",
    fontsize=6, markerscale=3, ncol=1,
    framealpha=0.9
)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "figure5_de_moa.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {out_path}")

print(f"\nDONE")