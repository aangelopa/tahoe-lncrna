"""
PC1 Fork Investigation
=========================
Follow-up to pc1_magnitude_check.py.

This script:
  1. Recomputes the correlation using |PC1| instead of signed PC1 - this
     should be much higher if magnitude drives the SIZE of PC1 regardless
     of its sign/branch.
  2. Recolors the PC1-vs-magnitude scatter by Leiden cluster and by MOA,
     to test whether the two branches correspond to two distinct
     mechanistic/cluster groups (interesting, real biology) rather than
     a technical artifact.

Requires the same files as pc1_magnitude_check.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURATION - same as pc1_magnitude_check.py
# ─────────────────────────────────────────────
OUTPUT_DIR = "/home/a/aangelopa/Thesis/Data/2ndtry/drug_level_leiden"

RAW_MATRIX_PATH     = f"{OUTPUT_DIR}/drug_level_matrix_raw.parquet"
ZSCORED_MATRIX_PATH = f"{OUTPUT_DIR}/drug_level_matrix_zscored.parquet"
META_PATH           = f"{OUTPUT_DIR}/drug_leiden_metadata.parquet"
MOA_COL             = "moa-fine"

RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading drug-level matrices and metadata...")
raw_matrix = pd.read_parquet(RAW_MATRIX_PATH)
z_matrix   = pd.read_parquet(ZSCORED_MATRIX_PATH)
meta       = pd.read_parquet(META_PATH)
meta       = meta.set_index("drug").loc[raw_matrix.index].reset_index()

# ─────────────────────────────────────────────
# 2. RECOMPUTE PC1 + MAGNITUDE
# ─────────────────────────────────────────────
print("Recomputing PCA and magnitude...")
pca = PCA(n_components=5, random_state=RANDOM_STATE)
pcs = pca.fit_transform(z_matrix.values)
pc1 = pcs[:, 0]

magnitude = np.sqrt((raw_matrix.values ** 2).sum(axis=1))
log_magnitude = np.log1p(magnitude)

# ─────────────────────────────────────────────
# 3. CORRELATION WITH |PC1|
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CORRELATION: |PC1| vs magnitude")
print("=" * 60)

rho_signed, p_signed = spearmanr(pc1, log_magnitude)
rho_abs, p_abs = spearmanr(np.abs(pc1), log_magnitude)

print(f"  Signed PC1 vs log1p(magnitude):  Spearman r = {rho_signed:.3f}  (p = {p_signed:.2e})")
print(f"  |PC1|   vs log1p(magnitude):      Spearman r = {rho_abs:.3f}  (p = {p_abs:.2e})")

if rho_abs > 0.8 and abs(rho_signed) < rho_abs - 0.2:
    print("\n  -> CONFIRMED: magnitude drives the SIZE of PC1 (how far from 0),")
    print("     not its direction. The two branches represent two distinct")
    print("     signatures that both scale with perturbation strength.")
elif rho_abs > 0.8:
    print("\n  -> |PC1| strongly tracks magnitude, similar to signed PC1 -")
    print("     the fork may be less pronounced than it visually appeared.")
else:
    print("\n  -> Neither signed nor |PC1| strongly tracks magnitude -")
    print("     something else may be driving PC1's structure. Worth")
    print("     checking PC1 loadings (which gene sets contribute most).")

# ─────────────────────────────────────────────
# 4. RECOLOR SCATTER BY LEIDEN CLUSTER AND MOA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Plotting PC1-vs-magnitude colored by Leiden cluster and MOA")
print("=" * 60)

def make_color_map(values):
    unique_vals = sorted(set(values), key=lambda v: str(v))
    all_colors = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
    ])
    color_map = {v: all_colors[i % len(all_colors)] for i, v in enumerate(unique_vals)}
    return unique_vals, color_map

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Colored by Leiden cluster ---
clusters = meta["leiden_drug"].values
unique_clusters, cluster_color = make_color_map(clusters)
for c in unique_clusters:
    idx = clusters == c
    axes[0].scatter(log_magnitude[idx], pc1[idx], s=20, alpha=0.7,
                     color=cluster_color[c], label=f"Cluster {c}")
axes[0].set_xlabel("log1p(L2-norm perturbation magnitude)")
axes[0].set_ylabel("PC1 (drug-level embedding)")
axes[0].set_title("PC1 vs Magnitude — colored by Leiden cluster")
axes[0].legend(fontsize=7, markerscale=1.3, loc="best")
axes[0].grid(alpha=0.3)
axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")

# --- Colored by MOA (only known MOAs, top N by frequency for readability) ---
moas = meta[MOA_COL].fillna("unknown").values
moa_counts = pd.Series(moas).value_counts()
top_moas = [m for m in moa_counts.index if m not in ("unclear", "unknown")][:12]

plotted_mask = np.isin(moas, top_moas)
axes[1].scatter(log_magnitude[~plotted_mask], pc1[~plotted_mask],
                 s=15, alpha=0.15, color="lightgray", label="other/unclear")

unique_moas, moa_color = make_color_map(top_moas)
for m in top_moas:
    idx = moas == m
    axes[1].scatter(log_magnitude[idx], pc1[idx], s=25, alpha=0.8,
                     color=moa_color[m], label=m)
axes[1].set_xlabel("log1p(L2-norm perturbation magnitude)")
axes[1].set_ylabel("PC1 (drug-level embedding)")
axes[1].set_title("PC1 vs Magnitude — colored by MOA (top 12 known)")
axes[1].legend(fontsize=6, markerscale=1.2, loc="best", ncol=1)
axes[1].grid(alpha=0.3)
axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")

plt.tight_layout()
out_path = f"{OUTPUT_DIR}/pc1_fork_by_cluster_and_moa.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")

# ─────────────────────────────────────────────
# 5. QUANTIFY: does branch (sign of PC1) align with cluster/MOA?
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Branch composition by Leiden cluster")
print("=" * 60)
branch = np.where(pc1 > 0, "positive", "negative")
branch_ct = pd.crosstab(meta["leiden_drug"], branch)
print(branch_ct)

print("\n" + "=" * 60)
print("Branch composition by top MOAs")
print("=" * 60)
meta_top_moa = meta[np.isin(moas, top_moas)].copy()
meta_top_moa["branch"] = branch[np.isin(moas, top_moas)]
branch_moa_ct = pd.crosstab(meta_top_moa[MOA_COL], meta_top_moa["branch"])
print(branch_moa_ct)