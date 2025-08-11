import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


class LayoutVisualizer:

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
    def visualize_layout_grid(df, cluster_labels=None):
        # 1. Đọc dữ liệu
        if df is None or df.empty:
            print("Dữ liệu rỗng.")
            return

        # 2. Ước lượng tham số layout
        coords = df[["x", "y"]].values
        dist_matrix = squareform(pdist(coords))
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = max(dist_matrix.min(), 50)
        dept_size = round(min_dist * 0.8)
        cell_size = max(1, round(dept_size / 2))
        padding = dept_size

        # 3. Tạo lưới từ dữ liệu xy
        categories = df["Category"].unique()
        dept_id_map = {name: i + 1 for i, name in enumerate(categories)}
        df["Dept_ID"] = df["Category"].map(dept_id_map)
        df.rename(columns={"x": "center_x_px", "y": "center_y_px"}, inplace=True)
        df["x_start"] = df["center_x_px"] - dept_size / 2
        df["x_end"] = df["center_x_px"] + dept_size / 2
        df["y_start"] = df["center_y_px"] - dept_size / 2
        df["y_end"] = df["center_y_px"] + dept_size / 2
        min_x, max_x = df["x_start"].min() - padding, df["x_end"].max() + padding
        min_y, max_y = df["y_start"].min() - padding, df["y_end"].max() + padding
        grid_w = math.ceil((max_x - min_x) / cell_size)
        grid_h = math.ceil((max_y - min_y) / cell_size)
        grid = np.zeros((grid_h, grid_w), dtype=int)
        for _, row in df.iterrows():
            x0 = math.floor((row["x_start"] - min_x) / cell_size)
            x1 = math.ceil((row["x_end"] - min_x) / cell_size)
            y0 = math.floor((row["y_start"] - min_y) / cell_size)
            y1 = math.ceil((row["y_end"] - min_y) / cell_size)
            grid[y0:y1, x0:x1] = row["Dept_ID"]

        # 4. Xuất file mapping Category - Dept_ID
        pd.DataFrame(list(dept_id_map.items()), columns=["Category", "Dept_ID"]).to_csv(
            "layout_department_id.csv", index=False
        )

        # 5. Trực quan hóa layout trên lưới
        fig, ax = plt.subplots(figsize=(15, 15 * (grid.shape[0] / grid.shape[1])))
        unique_ids = np.unique(grid)
        max_id = unique_ids[unique_ids > 0].max() if (unique_ids > 0).any() else 0
        colors = ["white"] + [plt.cm.tab20(i / 20) for i in range(int(max_id))]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(list(range(0, int(max_id) + 2)), cmap.N)
        ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
        id_to_name = {v: k for k, v in dept_id_map.items()}
        for dept_id in unique_ids[unique_ids > 0]:
            ys, xs = np.where(grid == dept_id)
            if len(xs) > 0:
                cx, cy = np.mean(xs), np.mean(ys)
                name = id_to_name.get(dept_id, f"ID {dept_id}")
                col = cmap(norm(dept_id))
                luminance = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                tcolor = "white" if luminance < 0.5 else "black"
                rot = (
                    90
                    if (ys.max() - ys.min() + 1) > 1.5 * (xs.max() - xs.min() + 1)
                    else 0
                )
                ax.text(
                    cx,
                    cy,
                    name,
                    va="center",
                    ha="center",
                    color=tcolor,
                    fontsize=10,
                    rotation=rot,
                )
        ax.set_title("Trực quan hóa Layout trên Lưới", fontsize=16)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        ax.tick_params(
            axis="both",
            which="major",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
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
