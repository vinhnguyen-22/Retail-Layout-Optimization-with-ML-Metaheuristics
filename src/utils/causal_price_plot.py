# plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_causal_heatmap(df_results, segment="mid", save_path=None):
    df_seg = df_results[df_results["segment"] == segment]
    if df_seg.empty:
        print("No data to plot!")
        return
    pivot = df_seg.pivot(index="treatment", columns="target", values="mean_cf_effect")
    plt.figure(figsize=(1 + 0.5 * len(pivot.columns), 1 + 0.4 * len(pivot.index)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Causal Effect"},
    )
    plt.title(
        f"Causal Effect Heatmap (segment: {segment})\nRed: Bổ trợ, Blue: Thay thế"
    )
    plt.ylabel("Treatment (Price_X)")
    plt.xlabel("Target (Qty_Y)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bootstrap_boxplot(df_results, segment="mid", top_n=10, save_path=None):
    df_seg = df_results[df_results["segment"] == segment].copy()
    if df_seg.empty or "bootstrap_samples" not in df_seg.columns:
        print("No data to plot boxplot!")
        return
    df_seg["abs_effect"] = df_seg["mean_cf_effect"].abs()
    df_seg = df_seg.sort_values("abs_effect", ascending=False).head(top_n)
    plt.figure(figsize=(1.5 * top_n, 6))
    bplot_data = [row["bootstrap_samples"] for _, row in df_seg.iterrows()]
    labels = [f"{row['treatment']}→{row['target']}" for _, row in df_seg.iterrows()]
    sns.boxplot(data=bplot_data)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    plt.axhline(0, ls="--", color="red", alpha=0.5)
    plt.ylabel("Bootstrap Causal Effect")
    plt.title(f"Bootstrap Effect (Top {top_n}, segment={segment})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_scatter_estimators(df_results, segment="mid", save_path=None):
    df_seg = df_results[df_results["segment"] == segment]
    if df_seg.empty:
        print("No data to plot scatterplot!")
        return
    plt.figure(figsize=(8, 8))
    plt.scatter(df_seg["mean_dml_effect"], df_seg["mean_cf_effect"], alpha=0.7, s=70)
    plt.axhline(0, color="grey", ls="--", lw=1)
    plt.axvline(0, color="grey", ls="--", lw=1)
    plt.xlabel("Mean LinearDML Effect")
    plt.ylabel("Mean CausalForestDML Effect")
    plt.title(f"Scatter: CausalForestDML vs LinearDML (segment={segment})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_effect_distribution(df_results, segment="mid", save_path=None):
    df_seg = df_results[df_results["segment"] == segment]
    if df_seg.empty:
        print("No data to plot effect distribution!")
        return
    plt.figure(figsize=(9, 5))
    sns.histplot(df_seg["mean_cf_effect"], kde=True, bins=15)
    plt.axvline(0, color="red", ls="--", lw=1)
    plt.xlabel("Mean CausalForestDML Effect")
    plt.title(f"Distribution of Causal Effect (segment={segment})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
