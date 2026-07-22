"""
Corrected sub-cluster MOA breakdown — restricted to old cluster 7 only.
===========================================================================
Bug in the previous run: the MOA breakdown per sub-cluster was computed
over the ENTIRE new cluster (all comparisons that landed there, regardless
of which original cluster they came from), not restricted to the subset
that came specifically from old cluster 7. This fixes that.

Requires: comparison_leiden_metadata_with_highres.parquet, already saved
by comparison_followup_checks.py — no need to rerun PCA/Leiden.
"""

import pandas as pd

OUTPUT_DIR = "/home/a/aangelopa/Thesis/Data/2ndtry/comparison_level_full"
MOA_COL = "moa-fine"
KEY_CLUSTER = "7"

meta = pd.read_parquet(f"{OUTPUT_DIR}/comparison_leiden_metadata_with_highres.parquet")

# THE FIX: restrict to comparisons that were in OLD cluster 7 first,
# THEN break down by which NEW cluster they landed in
old_cluster_mask = meta["leiden_comp"].astype(str) == str(KEY_CLUSTER)
meta_old7 = meta[old_cluster_mask].copy()

print(f"Original cluster {KEY_CLUSTER} size: {len(meta_old7):,}")
print()

# Only look at the new clusters that actually received a meaningful
# number of old-cluster-7 comparisons (skip the near-zero ones)
redistribution = meta_old7["leiden_high_res"].value_counts()
meaningful_destinations = redistribution[redistribution >= 20].index.tolist()

for new_cluster in meaningful_destinations:
    # Restrict to: old cluster 7 AND this specific new cluster
    subset = meta_old7[meta_old7["leiden_high_res"] == new_cluster]
    subset_known = subset[subset[MOA_COL].notna() & (subset[MOA_COL] != "unclear")]

    n_total = len(subset)
    n_known = len(subset_known)
    print(f"New cluster {new_cluster}  (n={n_total} from old cluster {KEY_CLUSTER}, "
          f"{n_known} with known MOA):")
    if n_known > 0:
        top_moas = subset_known[MOA_COL].value_counts(normalize=True).head(6)
        for moa, frac in top_moas.items():
            count = subset_known[MOA_COL].value_counts()[moa]
            print(f"    {moa:<35} {count:>5}  ({frac:.1%} of known-MOA in this sub-group)")
    print()