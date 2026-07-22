"""
Merge E-distance results and plot Figure 5B/C
==============================================
Run after all E-distance array jobs complete.

Loads:
  - E-distance parquets (per plate)
  - tsne_embedding.npy
  - nnmds_embedding.npy
  - metadata.parquet

Outputs:
  - e_distance_all.parquet (merged E-distances)
  - figure5_bc.pdf (t-SNE and NNMDS colored by E-distance)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
E_DIST_DIR = os.path.expanduser("~/Thesis/Data/e_distance/")
EMBED_DIR  = os.path.expanduser("~/Thesis/Data/dim_reduction/")
OUTPUT_DIR = EMBED_DIR

# ─────────────────────────────────────────────
# 1. MERGE E-DISTANCE FILES
# ─────────────────────────────────────────────
print("1. Merging E-distance parquets...")

edist_files = sorted(glob.glob(os.path.join(E_DIST_DIR, "e_distance_*.parquet")))
print(f"   Found {len(edist_files)} files")

edist_df = pd.concat([pd.read_parquet(f) for f in edist_files], axis=0)
print(f"   Total E-distance entries: {len(edist_df)}")

# Save merged
edist_df.to_parquet(os.path.join(OUTPUT_DIR, "e_distance_all.parquet"), index=False)

# ─────────────────────────────────────────────
# 2. LOAD EMBEDDINGS AND METADATA
# ─────────────────────────────────────────────
print("\n2. Loading embeddings and metadata...")

tsne = np.load(os.path.join(EMBED_DIR, "tsne_embedding.npy"))
nnmds = np.load(os.path.join(EMBED_DIR, "nnmds_embedding.npy"))
metadata = pd.read_parquet(os.path.join(EMBED_DIR, "metadata.parquet"))

print(f"   t-SNE shape:  {tsne.shape}")
print(f"   NNMDS shape:  {nnmds.shape}")
print(f"   Metadata:     {metadata.shape}")

# ─────────────────────────────────────────────
# 3. MATCH E-DISTANCES TO EMBEDDINGS
# ─────────────────────────────────────────────
print("\n3. Matching E-distances to embedding rows...")

# Build comparison_id from metadata (same format as E-distance output)
metadata["comparison_id"] = (
    metadata["drug"] + " | " + metadata["plate"] + " | " + metadata["cell_line"]
)

# Create E-distance lookup
edist_lookup = edist_df.set_index("comparison_id")["log1p_e_distance"].to_dict()

# Map to metadata
metadata["log1p_e_distance"] = metadata["comparison_id"].map(edist_lookup)
metadata["e_distance"] = metadata["comparison_id"].map(
    edist_df.set_index("comparison_id")["e_distance"].to_dict()
)

n_matched = metadata["log1p_e_distance"].notna().sum()
n_total = len(metadata)
print(f"   Matched: {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)")

if n_matched == 0:
    print("   ERROR: No matches found! Check comparison_id format.")
    print(f"   Sample metadata IDs: {metadata['comparison_id'].head(3).tolist()}")
    print(f"   Sample E-dist IDs:   {edist_df['comparison_id'].head(3).tolist()}")
    exit(1)

# ─────────────────────────────────────────────
# 4. PLOT FIGURE 5B/C
# ─────────────────────────────────────────────
print("\n4. Generating Figure 5B/C...")

# Use log1p E-distance for coloring (as in paper)
e_vals = metadata["log1p_e_distance"].values

# For points without E-distance, use gray
has_edist = ~np.isnan(e_vals)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, embedding, title in [
    (axes[0], tsne, "t-SNE"),
    (axes[1], nnmds, "NNMDS"),
]:
    # Plot points without E-distance in gray first
    if (~has_edist).any():
        ax.scatter(
            embedding[~has_edist, 0], embedding[~has_edist, 1],
            c="lightgray", s=1, alpha=0.2, rasterized=True, zorder=1
        )

    # Plot points with E-distance, colored by value
    if has_edist.any():
        sc = ax.scatter(
            embedding[has_edist, 0], embedding[has_edist, 1],
            c=e_vals[has_edist], cmap="viridis", s=2, alpha=0.6,
            rasterized=True, zorder=2,
            vmin=np.nanpercentile(e_vals[has_edist], 2),
            vmax=np.nanpercentile(e_vals[has_edist], 98),
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("log₁₊ E-distance", fontsize=10)

    ax.set_title(f"{title} colored by E-distance", fontsize=13)
    ax.set_xlabel(f"{title} 1")
    ax.set_ylabel(f"{title} 2")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "figure5_bc.pdf")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {out_path}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("DONE")
print(f"  E-distance entries: {len(edist_df)}")
print(f"  Matched to embeddings: {n_matched}/{n_total}")
print(f"  E-distance stats (log1p):")
print(f"    mean:   {np.nanmean(e_vals):.4f}")
print(f"    median: {np.nanmedian(e_vals):.4f}")
print(f"    max:    {np.nanmax(e_vals[has_edist]):.4f}")
print(f"{'=' * 60}")