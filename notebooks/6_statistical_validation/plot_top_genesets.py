"""
Diverging Bar Chart — Top N Over/Under-expressed Gene Sets per Drug
====================================================================
Reads the CSV output of top_genesets_per_drug.py and produces a
multi-panel figure: one diverging bar chart per drug.

Each panel shows:
  - Top N gene sets with highest diff Vision score (right, coral/red)
  - Top N gene sets with lowest diff Vision score (left, teal/blue)
  - Bars sorted by absolute score magnitude (strongest at top)
  - Score annotated on each bar

Output:
  - PDF:  top_genesets_diverging.pdf   (vector, publication-quality)
  - PNG:  top_genesets_diverging.png   (raster, 150 dpi)


"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CSV     = Path.home() / "Thesis" / "Results" / "top_genesets" / "top_genesets_per_drug.csv"
DEFAULT_OUT_DIR = Path.home() / "Thesis" / "Results" / "top_genesets"
DEFAULT_NCOLS   = 3
DEFAULT_FONT    = 8       # pt — compact so gene set names fit
PANEL_W         = 5.5     # inches per panel
PANEL_H_PER_BAR = 0.32    # inches per bar row
PANEL_H_MIN     = 3.5     # minimum panel height in inches

# Colors (colorblind-friendly, print-safe)
COLOR_OVER  = "#C0392B"   # deep red  — overexpressed
COLOR_UNDER = "#1A5276"   # deep blue — underexpressed
COLOR_ZERO  = "#BDC3C7"   # light gray zero line


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def shorten_name(name, max_chars=52):
    """Trim long gene set names with ellipsis."""
    return name if len(name) <= max_chars else name[:max_chars - 1] + "…"


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    required = {"drug", "direction", "rank", "gene_set", "score"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV missing columns: {missing}")
    return df


def draw_panel(ax, drug_name, over_df, under_df, fontsize, n):
    """
    Draw one diverging bar chart panel for a single drug.

    over_df  : DataFrame with columns [gene_set, score], top N overexpressed
    under_df : DataFrame with columns [gene_set, score], top N underexpressed
    """
    # Combine: overexpressed at top, underexpressed below, separated by a gap
    # Sort each group by |score| descending (strongest bar at the inner edge)
    over_sorted  = over_df.sort_values("score", ascending=True)   # plot bottom→top
    under_sorted = under_df.sort_values("score", ascending=False)  # plot bottom→top

    labels = (
        [shorten_name(g) for g in under_sorted["gene_set"]]
        + [""]   # blank separator row
        + [shorten_name(g) for g in over_sorted["gene_set"]]
    )
    scores = (
        list(under_sorted["score"])
        + [0.0]
        + list(over_sorted["score"])
    )
    colors = (
        [COLOR_UNDER] * len(under_sorted)
        + [COLOR_ZERO]
        + [COLOR_OVER]  * len(over_sorted)
    )

    y_pos = np.arange(len(labels))

    bars = ax.barh(
        y_pos, scores, color=colors,
        height=0.65, edgecolor="none", zorder=2,
    )

    # Zero line
    ax.axvline(0, color=COLOR_ZERO, linewidth=0.8, zorder=1)

    # Score annotations on bars
    for i, (score, color) in enumerate(zip(scores, colors)):
        if color == COLOR_ZERO:
            continue
        # Position label just outside the bar end
        offset = 0.002 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if ax.get_xlim()[1] != 0 else 0.001
        ha     = "left"  if score >= 0 else "right"
        x_pos  = score + (0.003 if score >= 0 else -0.003)
        ax.text(
            x_pos, y_pos[i], f"{score:+.3f}",
            va="center", ha=ha, fontsize=fontsize - 1.5,
            color=color,
        )

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=fontsize)

    # Tick params
    ax.tick_params(axis="x", labelsize=fontsize - 0.5)
    ax.tick_params(axis="y", length=0, pad=4)

    # Remove spines
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("#AAAAAA")

    # Grid
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(axis="x", which="major", linestyle="--", linewidth=0.4,
            color="#DDDDDD", zorder=0)

    # X label
    ax.set_xlabel("diff Vision score (drug − DMSO)", fontsize=fontsize - 0.5,
                  color="#555555")

    # Panel title
    ax.set_title(drug_name, fontsize=fontsize + 1, fontweight="bold",
                 loc="left", pad=6)

    # Legend inside panel (top-right corner)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_OVER,  label="overexpressed"),
        Patch(facecolor=COLOR_UNDER, label="underexpressed"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=fontsize - 1.5,
        loc="lower right",
        frameon=True,
        framealpha=0.8,
        edgecolor="#DDDDDD",
        handlelength=1.0,
        handletextpad=0.4,
    )

    # Extend x limits slightly so score labels don't clip
    xmin, xmax = ax.get_xlim()
    pad = (xmax - xmin) * 0.18
    ax.set_xlim(xmin - pad, xmax + pad)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diverging bar chart of top N gene sets per drug.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",     type=str, default=str(DEFAULT_CSV),
                        help="Path to top_genesets_per_drug.csv")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Output directory for figures")
    parser.add_argument("--drugs",   nargs="+", default=None,
                        help="Restrict to specific drugs")
    parser.add_argument("--ncols",   type=int, default=DEFAULT_NCOLS,
                        help=f"Panels per row (default: {DEFAULT_NCOLS})")
    parser.add_argument("--fontsize",type=float, default=DEFAULT_FONT,
                        help=f"Base font size in pt (default: {DEFAULT_FONT})")
    parser.add_argument("--dpi",     type=int, default=150,
                        help="PNG resolution (default: 150)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading: {args.csv}")
    df = load_results(args.csv)

    drugs = df["drug"].unique()
    if args.drugs:
        drugs = [d for d in drugs if d in args.drugs]
        if not drugs:
            sys.exit(f"ERROR: None of the requested drugs found in CSV.")
    drugs = sorted(drugs)

    # Infer N from data
    n = df[df["direction"] == "overexpressed"]["rank"].max()
    print(f"  Drugs: {len(drugs)}, N per direction: {n}")

    # ── Layout ────────────────────────────────────────────────────────────────
    ncols   = min(args.ncols, len(drugs))
    nrows   = math.ceil(len(drugs) / ncols)
    n_bars  = 2 * n + 1  # over + separator + under
    panel_h = max(PANEL_H_MIN, n_bars * PANEL_H_PER_BAR + 1.0)

    fig_w = PANEL_W * ncols
    fig_h = panel_h * nrows

    print(f"  Layout: {nrows} rows × {ncols} cols, "
          f"figure {fig_w:.1f}″ × {fig_h:.1f}″")

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    plt.rcParams.update({
        "font.family":  "DejaVu Sans",
        "font.size":    args.fontsize,
        "axes.unicode_minus": False,
    })

    # ── Draw panels ───────────────────────────────────────────────────────────
    for idx, drug in enumerate(drugs):
        row = idx // ncols
        col = idx % ncols
        ax  = axes[row][col]

        sub      = df[df["drug"] == drug]
        over_df  = sub[sub["direction"] == "overexpressed"][["gene_set", "score"]]
        under_df = sub[sub["direction"] == "underexpressed"][["gene_set", "score"]]

        draw_panel(ax, drug, over_df, under_df, args.fontsize, n)

    # Hide unused panels
    for idx in range(len(drugs), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].set_visible(False)

    # ── Figure-level title ────────────────────────────────────────────────────
    fig.suptitle(
        f"Top {n} over- and under-expressed gene sets per drug\n"
        f"(median differential Vision score across cell lines and plates)",
        fontsize=args.fontsize + 2,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(rect=[0, 0, 1, 1], h_pad=2.5, w_pad=2.0)

    # ── Save ──────────────────────────────────────────────────────────────────
    pdf_path = out_dir / "top_genesets_diverging.pdf"
    png_path = out_dir / "top_genesets_diverging.png"

    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    print(f"Saved PDF → {pdf_path}")

    fig.savefig(png_path, bbox_inches="tight", dpi=args.dpi)
    print(f"Saved PNG → {png_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()