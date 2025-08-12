from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from src.utils.plot_utils import (
    DEFAULT_CMAP_NAME,
    DEFAULT_DPI,
    DEFAULT_PADDING_RATIO,
    _axes_off,
    _build_shared_mapping,
    _draw_grid,
    _draw_labels,
    _estimate_cell_size,
    _figure_size,
    _make_cmap_norm,
    _rasterize_grid,
    _validate_df,
)


class LayoutVisualizer:

    @staticmethod
    def plot_visualize_layout(
        self,
        df_layout: pd.DataFrame,
        out_png: Path,
        cell_size: Optional[float] = None,
        title: str = "Layout (GA) — preview",
        label_fontsize: int = 4,
        show_labels: bool = True,
    ) -> None:
        _validate_df(df_layout)
        cs = (
            float(cell_size)
            if cell_size is not None
            else _estimate_cell_size(df_layout)
        )
        grid, id2name = _rasterize_grid(
            df_layout, cs, padding_ratio=DEFAULT_PADDING_RATIO
        )
        _draw_grid(
            grid,
            id2name,
            title=title,
            out_png=out_png,
            label_fontsize=(label_fontsize if show_labels else 0),
            dpi=DEFAULT_DPI,
            cmap_name=DEFAULT_CMAP_NAME,
        )

    def plot_compare_layouts(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        out_png: Path,
        titles: Tuple[str, str] = ("Trước", "Sau"),
        cell_size_before: Optional[float] = None,
        cell_size_after: Optional[float] = None,
        label_mode: str = "both",  # "left" | "right" | "both" | "none"
        label_fontsize: int = 4,
    ) -> None:
        _validate_df(df_before)
        _validate_df(df_after)

        shared_map = _build_shared_mapping(df_before, df_after)
        cs_a = (
            float(cell_size_before)
            if cell_size_before is not None
            else _estimate_cell_size(df_before)
        )
        cs_b = (
            float(cell_size_after)
            if cell_size_after is not None
            else _estimate_cell_size(df_after)
        )

        grid_a, id2name = _rasterize_grid(
            df_before, cs_a, name2id=shared_map, padding_ratio=DEFAULT_PADDING_RATIO
        )
        grid_b, _ = _rasterize_grid(
            df_after, cs_b, name2id=shared_map, padding_ratio=DEFAULT_PADDING_RATIO
        )

        H1, W1 = int(grid_a.shape[0]), int(grid_a.shape[1])
        H2, W2 = int(grid_b.shape[0]), int(grid_b.shape[1])
        max_id = max(
            int(grid_a.max()) if grid_a.size else 0,
            int(grid_b.max()) if grid_b.size else 0,
        )
        cmap, norm = _make_cmap_norm(max_id, DEFAULT_CMAP_NAME)

        fig_w, fig_h = _figure_size(W1 + W2, max(H1, H2), cols=2)
        fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h))

        axs[0].imshow(grid_a, cmap=cmap, norm=norm, interpolation="none")
        axs[0].set_title(titles[0], fontsize=6)
        _axes_off(axs[0])

        axs[1].imshow(grid_b, cmap=cmap, norm=norm, interpolation="none")
        axs[1].set_title(titles[1], fontsize=6)
        _axes_off(axs[1])

        if label_mode in ("left", "both"):
            _draw_labels(axs[0], grid_a, id2name, fontsize=label_fontsize)
        if label_mode in ("right", "both"):
            _draw_labels(axs[1], grid_b, id2name, fontsize=label_fontsize)

        plt.savefig(out_png, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_spring_layout(
        affinity_matrix, threshold=0.0, cluster_labels=None, node_size=1000
    ):
        """
        Vẽ spring layout của network ngành hàng theo affinity_matrix.
        Các cạnh có trọng số > threshold sẽ được vẽ.
        Nếu có cluster_labels, sẽ tô màu theo cluster.
        """
        G = nx.Graph()
        cats = list(affinity_matrix.index)
        G.add_nodes_from(cats)

        # Thêm các cạnh theo affinity > threshold
        for i, row in affinity_matrix.iterrows():
            for j, v in row.items():
                if i != j and v > threshold:
                    G.add_edge(i, j, weight=v)
        pos = nx.spring_layout(G, k=1.0, iterations=70, seed=42)
        weights = [G[u][v]["weight"] for u, v in G.edges()]

        # Xác định màu node
        if cluster_labels is not None:
            from matplotlib import cm

            cmap = cm.get_cmap("tab10", np.max(cluster_labels) + 1)
            color_map = [cmap(cluster_labels[cats.index(node)]) for node in G.nodes()]
        else:
            color_map = "#84C1FF"

        plt.figure(figsize=(13, 9))
        nx.draw_networkx_nodes(
            G, pos, node_color=color_map, node_size=node_size, alpha=0.9
        )
        nx.draw_networkx_edges(G, pos, width=[w * 2 for w in weights], alpha=0.35)
        nx.draw_networkx_labels(G, pos, font_size=12)
        plt.title("Spring layout - Network of Category Affinity")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_affinity_heatmap(affinity_matrix, title="Affinity Heatmap"):
        """
        Plot heatmap của affinity matrix (Pandas DataFrame, index và columns là category).
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(affinity_matrix, cmap="YlGnBu")
        plt.title(title)
        plt.xlabel("To")
        plt.ylabel("From")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_affinity_bar(affinity_matrix):
        """
        Plot tổng affinity mỗi ngành hàng (bar chart).
        """
        sums = affinity_matrix.sum(axis=1).sort_values(ascending=False)
        plt.figure(figsize=(14, 4))
        sums.plot(kind="bar")
        plt.title("Total Affinity by Category")
        plt.ylabel("Sum Affinity")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_ga_convergence(logbook):
        """
        Plot convergence GA (logbook là list of dict từ GA optimizer).
        """
        log_df = pd.DataFrame(logbook)
        plt.figure(figsize=(8, 4))
        if "max" in log_df:
            plt.plot(log_df["max"], label="Max fitness")
        if "avg" in log_df:
            plt.plot(log_df["avg"], label="Avg fitness")
        if "min" in log_df:
            plt.plot(log_df["min"], label="Min fitness", linestyle="--", alpha=0.6)
        if "diversity" in log_df:
            plt.plot(
                log_df["diversity"], label="Diversity", color="green", linestyle=":"
            )
        plt.xlabel("Generation")
        plt.ylabel("Fitness / Diversity")
        plt.legend()
        plt.tight_layout()
        plt.title("GA Fitness & Diversity Over Generations")
        plt.show()

    @staticmethod
    def plot_optuna_trials(study):
        """
        Plot giá trị best fitness theo trial của Optuna.
        """
        df = study.trials_dataframe()
        plt.figure(figsize=(8, 4))
        plt.plot(df["value"], marker="o")
        plt.xlabel("Trial")
        plt.ylabel("Fitness")
        plt.title("Optuna Optimization Progress")
        plt.show()
        plt.figure(figsize=(8, 4))
        plt.plot(df["value"], marker="o")
        plt.xlabel("Trial")
        plt.ylabel("Fitness")
        plt.title("Optuna Optimization Progress")
        plt.show()
