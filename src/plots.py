import math
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
    MAX_CELLS,
    _axes_off,
    _build_shared_mapping,
    _draw_labels,
    _figure_size,
    _make_cmap_norm,
    _rasterize_grid,
    _validate_df,
)


class LayoutVisualizer:
    from pathlib import Path


from typing import Dict, List, Optional

import pandas as pd


class LayoutVisualizer:

    @staticmethod
    def plot_visualize_layout_plotly(
        df_layout: pd.DataFrame,
        out_html: Path,
        *,
        cmap_name: str = "tab20",  # có thể ghép: "tab20|tab20b|tab20c"
        name2id: Optional[
            Dict[str, int]
        ] = None,  # mapping Category -> id; nếu None sẽ thử _build_shared_mapping rồi fallback
        show_labels: bool = True,  # 1 nhãn/Category (tránh đè)
        label_fontsize: int = 9,
        padding_ratio: float = 0.02,
        min_label_area: float = 6.0,  # ẩn nhãn nếu tổng diện tích Category quá nhỏ
        dedupe_geometry: bool = True,  # bỏ slot trùng (x,y,w,h,Category)
        show_legend: bool = True,
        title: str = "Layout (GA) — interactive",
    ) -> None:
        import numpy as np
        import plotly.graph_objects as go
        from matplotlib import cm

        # ---- validate ----
        need_cols = {"Category", "x", "y", "width", "height"}
        missing = need_cols - set(df_layout.columns)
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc trong layout: {sorted(missing)}")
        if df_layout.empty:
            raise ValueError("DataFrame layout rỗng.")

        # ---- clean copy & types ----
        df = df_layout.copy()
        df["Category"] = (
            df["Category"].astype(str).str.strip()
        )  # tránh lệch màu do khoảng trắng
        for c in ["x", "y", "width", "height"]:
            df[c] = df[c].astype(float)
        if not {"cx", "cy"}.issubset(df.columns):
            df["cx"] = df["x"] + df["width"] / 2.0
            df["cy"] = df["y"] + df["height"] / 2.0

        # ---- bounds + padding ----
        x0, y0 = float(df["x"].min()), float(df["y"].min())
        x1 = float((df["x"] + df["width"]).max())
        y1 = float((df["y"] + df["height"]).max())
        pad_x = (x1 - x0) * float(padding_ratio)
        pad_y = (y1 - y0) * float(padding_ratio)
        min_x, min_y = x0 - pad_x, y0 - pad_y
        max_x, max_y = x1 + pad_x, y1 + pad_y

        # ---- stable mapping Category -> id (giống matplotlib) ----
        if name2id is None:
            try:
                # nếu project có helper này, dùng để giữ ID/màu y hệt như bản matplotlib
                name2id = _build_shared_mapping(df)  # type: ignore[name-defined]
            except Exception:
                cats_sorted = sorted(df["Category"].unique())
                name2id = {c: i + 1 for i, c in enumerate(cats_sorted)}
        max_id = max(name2id.values()) if name2id else 1

        # ---- DISCRETE palette (không nội suy) ----
        # cho phép ghép nhiều cmap bằng dấu |
        def _collect_listed_colors(spec: str):
            parts = [s.strip() for s in spec.split("|")]
            colors = []
            for p in parts:
                cmap = cm.get_cmap(p)
                if hasattr(cmap, "colors"):  # ListedColormap (vd: tab20)
                    colors.extend(list(cmap.colors))
                else:  # continuous -> sample rời rạc 12 điểm
                    colors.extend([cmap(i / 11.0) for i in range(12)])
            return colors or list(cm.get_cmap("tab20").colors)

        base_colors = _collect_listed_colors(cmap_name)
        L = len(base_colors)

        def _rgba_str(rgba, alpha=1.0):
            r, g, b = rgba[:3]
            return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"

        # map màu rời rạc theo id (id-1) % L → giống cách vẽ discrete trong matplotlib
        color_map: Dict[str, str] = {
            cat: _rgba_str(base_colors[(int(idx) - 1) % L], 1.0)
            for cat, idx in name2id.items()
        }

        # ---- dedupe hình học để tránh đúp hover/label ----
        df_draw = (
            df.drop_duplicates(subset=["x", "y", "width", "height", "Category"])
            if dedupe_geometry
            else df
        )

        # ---- shapes (tô màu theo colormap) ----
        shapes = []
        for r in df_draw.itertuples(index=False):
            if r.width <= 0 or r.height <= 0:
                continue
            shapes.append(
                dict(
                    type="rect",
                    x0=float(r.x),
                    y0=float(r.y),
                    x1=float(r.x + r.width),
                    y1=float(r.y + r.height),
                    line=dict(color="rgba(0,0,0,0.25)", width=1),
                    fillcolor=color_map.get(r.Category, "rgba(204,204,204,1.0)"),
                    layer="below",
                )
            )

        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(size=14), x=0.05),
            shapes=shapes,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        # Khoá tỉ lệ 1:1 và ĐẢO trục Y (không bị lật)
        fig.update_xaxes(
            range=[min_x, max_x],
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        )
        fig.update_yaxes(
            range=[max_y, min_y], visible=False, showgrid=False, zeroline=False
        )

        # ---- Hover từng slot: marker "vô hình" đặt ở tâm ô ----
        hoverdata = np.stack(
            [
                df_draw["Category"].values.astype(str),
                df_draw["x"].values.astype(float),
                df_draw["y"].values.astype(float),
                df_draw["width"].values.astype(float),
                df_draw["height"].values.astype(float),
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_draw["cx"],
                y=df_draw["cy"],
                mode="markers",
                marker=dict(size=10, opacity=0),
                customdata=hoverdata,
                hovertemplate=(
                    "Category: %{customdata[0]}<br>"
                    "x: %{customdata[1]:.2f} | y: %{customdata[2]:.2f}<br>"
                    "w: %{customdata[3]:.2f} | h: %{customdata[4]:.2f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.update_layout(hovermode="closest")

        # ---- 1 nhãn / Category (tâm trọng số diện tích) ----
        if show_labels:
            df["area"] = df["width"] * df["height"]
            g = (
                df.groupby("Category", as_index=False)
                .apply(
                    lambda d: pd.Series(
                        {
                            "cx": (d["cx"] * d["area"]).sum()
                            / max(d["area"].sum(), 1e-9),
                            "cy": (d["cy"] * d["area"]).sum()
                            / max(d["area"].sum(), 1e-9),
                            "sum_area": d["area"].sum(),
                        }
                    )
                )
                .reset_index(drop=True)
            )
            g = g[g["sum_area"] >= float(min_label_area)]
            fig.add_trace(
                go.Scatter(
                    x=g["cx"],
                    y=g["cy"],
                    mode="text",
                    text=g["Category"],
                    textposition="middle center",
                    textfont=dict(size=int(label_fontsize), color="black"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # ---- legend màu ổn định (shapes không lên legend) ----
        if show_legend:
            for cat in sorted(name2id.keys(), key=lambda c: name2id[c]):
                fig.add_trace(
                    go.Scatter(
                        x=[min_x],
                        y=[min_y],
                        mode="markers",
                        marker=dict(size=8, color=color_map[cat]),
                        name=cat,
                        hoverinfo="skip",
                        visible="legendonly",
                    )
                )

        out_html = Path(out_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)

    @staticmethod
    def plot_visualize_layout(
        df_layout: pd.DataFrame,
        out_png: Path,
        cell_size: Optional[float] = None,
        title: str = "Layout (GA) — preview",
        label_fontsize: int = 4,
        show_labels: bool = True,
    ) -> None:
        import math

        import matplotlib.pyplot as plt
        import numpy as np

        need_cols = {"Category", "x", "y", "width", "height"}
        missing = need_cols - set(df_layout.columns)
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc trong layout: {sorted(missing)}")
        if df_layout.empty:
            raise ValueError("DataFrame layout rỗng.")

        # --- ƯỚC LƯỢNG cell size ---
        def _estimate_cell_size(df: pd.DataFrame) -> float:
            w = df["width"].to_numpy(dtype=float)
            h = df["height"].to_numpy(dtype=float)
            min_dim = np.minimum(w, h)
            pos = min_dim[min_dim > 0]
            if pos.size == 0:
                return 5.0
            # lấy ~1/4 median, tối thiểu 1.0
            return float(max(1.0, math.floor(np.median(pos) / 4.0)))

        # Snap nhẹ để tránh số lẻ khó chia lưới (không bắt buộc)
        def _snap(df: pd.DataFrame, unit: float = 1.0) -> pd.DataFrame:
            out = df.copy()
            for c in ["x", "y", "width", "height"]:
                out[c] = (out[c].astype(float) / unit).round().astype(float) * unit
            return out

        df = _snap(df_layout, unit=1.0)

        cs = float(cell_size) if cell_size is not None else _estimate_cell_size(df)
        if cs <= 0:
            raise ValueError("cell_size phải > 0.")

        # --- mapping tên <-> id ---
        name2id = _build_shared_mapping(df)  # bạn đã có helper này
        id2name = {v: k for k, v in name2id.items()}

        # --- biên vẽ + padding ---
        x0, y0 = float(df["x"].min()), float(df["y"].min())
        x1 = float((df["x"] + df["width"]).max())
        y1 = float((df["y"] + df["height"]).max())
        pad_x = (x1 - x0) * DEFAULT_PADDING_RATIO
        pad_y = (y1 - y0) * DEFAULT_PADDING_RATIO
        min_x, min_y = x0 - pad_x, y0 - pad_y
        max_x, max_y = x1 + pad_x, y1 + pad_y

        W = int(math.ceil((max_x - min_x) / cs))
        H = int(math.ceil((max_y - min_y) / cs))

        # --- giới hạn kích thước ảnh ---
        if W * H > MAX_CELLS:
            scale = math.sqrt((W * H) / float(MAX_CELLS))
            cs *= scale
            W = int(math.ceil((max_x - min_x) / cs))
            H = int(math.ceil((max_y - min_y) / cs))

        # --- rasterize ---
        grid = np.zeros((H, W), dtype=np.int32)
        for _, r in df.iterrows():
            did = name2id.get(str(r["Category"]), 0)
            rx, ry = float(r["x"]), float(r["y"])
            rw, rh = float(r["width"]), float(r["height"])
            if rw <= 0 or rh <= 0:
                continue

            gx0 = int((rx - min_x) // cs)
            gy0 = int((ry - min_y) // cs)
            gx1 = int(math.ceil((rx + rw - min_x) / cs))
            gy1 = int(math.ceil((ry + rh - min_y) / cs))

            gx0 = max(0, min(gx0, W))
            gx1 = max(0, min(gx1, W))
            gy0 = max(0, min(gy0, H))
            gy1 = max(0, min(gy1, H))
            if gx0 >= gx1 or gy0 >= gy1:
                continue
            grid[gy0:gy1, gx0:gx1] = did

        max_id = int(grid.max()) if grid.size else 0
        cmap, norm = _make_cmap_norm(max_id, DEFAULT_CMAP_NAME)

        fig_w, fig_h = _figure_size(W, H, cols=1)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(grid, cmap=cmap, norm=norm, interpolation="none")

        if show_labels and label_fontsize > 0:
            _draw_labels(ax, grid, id2name, fontsize=label_fontsize)

        ax.set_title(title, fontsize=6)
        _axes_off(ax)
        plt.savefig(out_png, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_compare_layouts(
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

        def _estimate_cell_size(df: pd.DataFrame) -> float:
            min_dim = np.minimum(
                df["width"].to_numpy(dtype=float),
                df["height"].to_numpy(dtype=float),
            )
            pos = min_dim[min_dim > 0]
            if pos.size == 0:
                return 5.0
            return float(max(1.0, int(np.median(pos) / 4)))

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
        affinity_matrix,
        threshold=0.0,
        cluster_labels=None,
        node_size=1000,
        out_png: Optional[Path] = None,
    ):
        """
        Vẽ spring layout của network ngành hàng theo affinity_matrix.
        Các cạnh có trọng số > threshold sẽ được vẽ.
        Nếu có cluster_labels, sẽ tô màu theo cluster.
        Nếu out_png được cung cấp, sẽ lưu ảnh ra file.
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
        if out_png is not None:
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_affinity_heatmap(
        affinity_matrix, title="Affinity Heatmap", out_png: Optional[Path] = None
    ):
        """
        Plot heatmap của affinity matrix (Pandas DataFrame, index và columns là category).
        Nếu out_png được cung cấp, sẽ lưu ảnh ra file.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(affinity_matrix, cmap="YlGnBu")
        plt.title(title)
        plt.xlabel("To")
        plt.ylabel("From")
        plt.tight_layout()
        if out_png is not None:
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_affinity_bar(affinity_matrix, out_png: Optional[Path] = None):
        """
        Plot tổng affinity mỗi ngành hàng (bar chart).
        Nếu out_png được cung cấp, sẽ lưu ảnh ra file.
        """
        sums = affinity_matrix.sum(axis=1).sort_values(ascending=False)
        plt.figure(figsize=(14, 4))
        sums.plot(kind="bar")
        plt.title("Total Affinity by Category")
        plt.ylabel("Sum Affinity")
        plt.tight_layout()
        if out_png is not None:
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_ga_convergence(logbook, out_png: Optional[Path] = None):
        """
        Plot convergence GA (logbook là list of dict từ GA optimizer).
        Nếu out_png được cung cấp, sẽ lưu ảnh ra file.
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
        if out_png is not None:
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_optuna_trials(study, out_png: Optional[Path] = None):
        """
        Plot giá trị best fitness theo trial của Optuna.
        Nếu out_png được cung cấp, sẽ lưu ảnh ra file.
        """
        df = study.trials_dataframe()
        plt.figure(figsize=(8, 4))
        plt.plot(df["value"], marker="o")
        plt.xlabel("Trial")
        plt.ylabel("Fitness")
        plt.title("Optuna Optimization Progress")
        if out_png is not None:
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.show()
