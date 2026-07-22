"""
PC1 vs Magnitude Correlation Check
=====================================
Tests whether the dominant axis (PC1) of the drug-level embedding is
essentially just "how strongly did this drug perturb gene-set activity
overall" (the same L2-norm magnitude metric used in the E-distance
validation) - which would explain the single-gradient t-SNE shape rather
than genuine multi-cluster structure.

Uses the outputs already saved by drug_level_leiden.py:
  - drug_level_matrix_raw.parquet      (n_drugs x n_genesets, raw median scores)
  - drug_level_matrix_zscored.parquet  (same, z-scored per gene set)
  - drug_leiden_metadata.parquet       (drug, leiden_drug, TSNE1, TSNE2, MOA)

Output: prints Spearman correlations, saves a scatter plot.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURATION - update to match your OUTPUT_DIR from drug_level_leiden.py
# ─────────────────────────────────────────────
OUTPUT_DIR = "/home/aangelopa/Thesis/Data/2ndtry/drug_level_leiden"

RAW_MATRIX_PATH    = f"{OUTPUT_DIR}/drug_level_matrix_raw.parquet"
ZSCORED_MATRIX_PATH = f"{OUTPUT_DIR}/drug_level_matrix_zscored.parquet"
META_PATH          = f"{OUTPUT_DIR}/drug_leiden_metadata.parquet"

RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading drug-level matrices and metadata...")
raw_matrix = pd.read_parquet(RAW_MATRIX_PATH)          # n_drugs x n_genesets
z_matrix   = pd.read_parquet(ZSCORED_MATRIX_PATH)      # same, z-scored
meta       = pd.read_parquet(META_PATH)                # drug, leiden_drug, TSNE1, TSNE2, MOA

print(f"  Raw matrix    : {raw_matrix.shape}")
print(f"  Z-scored matrix: {z_matrix.shape}")
print(f"  Metadata      : {meta.shape}")

# Align indices (raw_matrix/z_matrix are indexed by drug name;
# meta has a 'drug' column with the same values)
meta = meta.set_index("drug").loc[raw_matrix.index].reset_index()

# ─────────────────────────────────────────────
# 2. RECOMPUTE PC1 FROM THE Z-SCORED MATRIX
#    (same input drug_level_leiden.py used for PCA -> Leiden -> t-SNE)
# ─────────────────────────────────────────────
print("\nRecomputing PCA on z-scored drug-level matrix...")
pca = PCA(n_components=5, random_state=RANDOM_STATE)
pcs = pca.fit_transform(z_matrix.values)
pc1 = pcs[:, 0]

ev = pca.explained_variance_ratio_
print(f"  Variance explained by PC1: {ev[0]*100:.1f}%")
print(f"  Variance explained by PC1-5: {ev.cumsum()[-1]*100:.1f}%")

# ─────────────────────────────────────────────
# 3. COMPUTE L2-NORM MAGNITUDE (same formula as your earlier validation)
#    magnitude = sqrt(sum(score_i^2)) across all gene sets, per drug
#    then log1p-transformed, exactly as before
# ─────────────────────────────────────────────
print("\nComputing L2-norm perturbation magnitude per drug...")
magnitude = np.sqrt((raw_matrix.values ** 2).sum(axis=1))
log_magnitude = np.log1p(magnitude)

# ─────────────────────────────────────────────
# 4. CORRELATE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CORRELATION RESULTS")
print("=" * 60)

rho_pc1, p_pc1 = spearmanr(pc1, log_magnitude)
print(f"  PC1 vs log1p(L2-norm magnitude):    Spearman r = {rho_pc1:.3f}  (p = {p_pc1:.2e})")

if "TSNE1" in meta.columns:
    rho_tsne1, p_tsne1 = spearmanr(meta["TSNE1"].values, log_magnitude)
    print(f"  TSNE1 vs log1p(L2-norm magnitude):  Spearman r = {rho_tsne1:.3f}  (p = {p_tsne1:.2e})")

print()
if abs(rho_pc1) > 0.8:
    print("  -> STRONG correlation: PC1 is essentially the magnitude axis.")
   
elif abs(rho_pc1) > 0.5:
    print("  -> MODERATE correlation: magnitude is a meaningful contributor")
    print("     to PC1, but other structure may also be present.")
else:
    print("  -> WEAK correlation: PC1 reflects something other than overall")
    print("     magnitude - the gradient shape has a different explanation.")

# ─────────────────────────────────────────────
# 5. PLOT FOR YOUR SLIDES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(log_magnitude, pc1, s=15, alpha=0.6)
axes[0].set_xlabel("log1p(L2-norm perturbation magnitude)")
axes[0].set_ylabel("PC1 (drug-level embedding)")
axes[0].set_title(f"PC1 vs Magnitude (Spearman r = {rho_pc1:.3f})")
axes[0].grid(alpha=0.3)

if "TSNE1" in meta.columns:
    axes[1].scatter(log_magnitude, meta["TSNE1"].values, s=15, alpha=0.6)
    axes[1].set_xlabel("log1p(L2-norm perturbation magnitude)")
    axes[1].set_ylabel("t-SNE 1 (drug-level embedding)")
    axes[1].set_title(f"t-SNE1 vs Magnitude (Spearman r = {rho_tsne1:.3f})")
    axes[1].grid(alpha=0.3)

plt.tight_layout()
out_path = f"{OUTPUT_DIR}/pc1_magnitude_correlation_check.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot -> {out_path}")