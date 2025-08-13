# src/plots_ga.py — GAPlotter chỉ còn 2 hàm public
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- module-level config ---
DEFAULT_PADDING_RATIO = 0.06
DEFAULT_DPI = 220
DEFAULT_CMAP_NAME = "tab20"
MAX_CELLS = 10_000_000


# ---------- helpers (module-level) ----------
def _validate_df(df: pd.DataFrame) -> None:
    need_cols = {"Category", "x", "y", "width", "height"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong layout: {sorted(missing)}")
    if df.empty:
        raise ValueError("DataFrame layout rỗng.")


def _build_shared_mapping(
    df_a: pd.DataFrame, df_b: Optional[pd.DataFrame] = None
) -> Dict[str, int]:
    cats_a = pd.Index(df_a["Category"].astype(str).unique())
    cats = (
        cats_a.union(pd.Index(df_b["Category"].astype(str).unique()))
        if df_b is not None
        else cats_a
    )
    return {c: i + 1 for i, c in enumerate(cats)}


def _rasterize_grid(
    df: pd.DataFrame,
    cell_size: float,
    name2id: Optional[Dict[str, int]] = None,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> Tuple[np.ndarray, Dict[int, str]]:
    if cell_size <= 0:
        raise ValueError("cell_size phải > 0.")
    if name2id is None:
        name2id = _build_shared_mapping(df)
    id2name = {v: k for k, v in name2id.items()}

    x0, y0 = float(df["x"].min()), float(df["y"].min())
    x1 = float((df["x"] + df["width"]).max())
    y1 = float((df["y"] + df["height"]).max())
    pad_x = (x1 - x0) * padding_ratio
    pad_y = (y1 - y0) * padding_ratio
    min_x, min_y = x0 - pad_x, y0 - pad_y
    max_x, max_y = x1 + pad_x, y1 + pad_y

    W = int(math.ceil((max_x - min_x) / cell_size))
    H = int(math.ceil((max_y - min_y) / cell_size))

    if W * H > MAX_CELLS:
        scale = math.sqrt((W * H) / float(MAX_CELLS))
        cell_size *= scale
        W = int(math.ceil((max_x - min_x) / cell_size))
        H = int(math.ceil((max_y - min_y) / cell_size))

    grid = np.zeros((H, W), dtype=np.int32)

    for _, r in df.iterrows():
        did = name2id[str(r["Category"])]
        rx, ry = float(r["x"]), float(r["y"])
        rw, rh = float(r["width"]), float(r["height"])
        if rw <= 0 or rh <= 0:
            continue
        gx0 = int((rx - min_x) // cell_size)
        gy0 = int((ry - min_y) // cell_size)
        gx1 = int(math.ceil((rx + rw - min_x) / cell_size))
        gy1 = int(math.ceil((ry + rh - min_y) / cell_size))

        gx0 = max(0, min(gx0, W))
        gx1 = max(0, min(gx1, W))
        gy0 = max(0, min(gy0, H))
        gy1 = max(0, min(gy1, H))
        if gx0 >= gx1 or gy0 >= gy1:
            continue
        grid[gy0:gy1, gx0:gx1] = did

    return grid, id2name


def _make_cmap_norm(max_id: int, cmap_name: str = DEFAULT_CMAP_NAME):
    base = plt.cm.get_cmap(cmap_name, max(1, max_id))
    colors = ["#F9F9F9"] + [base(i) for i in range(max_id)]
    cmap = mcolors.ListedColormap(colors)
    bounds = list(range(0, max_id + 2))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _draw_labels(ax, grid: np.ndarray, id2name: Dict[int, str], fontsize: int = 4):
    ids = np.unique(grid)
    ids = ids[ids > 0]
    for did in ids:
        ys, xs = np.where(grid == did)
        if xs.size == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        name = id2name.get(int(did), f"ID {int(did)}")
        ax.text(cx, cy, name, va="center", ha="center", fontsize=fontsize)


def _axes_off(ax):
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )


def _figure_size(W: int, H: int, cols: int = 1) -> Tuple[float, float]:
    base_w = 18.0 * (float(W) / max(1.0, float(H)))
    fig_w = min(30.0, base_w if cols == 1 else base_w)
    fig_h = min(30.0, 18.0)
    return float(fig_w), float(fig_h)


def _draw_grid(
    grid: np.ndarray,
    id2name: Dict[int, str],
    title: str,
    out_png: Path,
    label_fontsize: int,
    dpi: int = DEFAULT_DPI,
    cmap_name: str = DEFAULT_CMAP_NAME,
) -> None:
    H, W = grid.shape
    max_id = int(grid.max()) if grid.size else 0
    cmap, norm = _make_cmap_norm(max_id, cmap_name)

    fig_w, fig_h = _figure_size(W, H, cols=1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="none")

    if label_fontsize > 0:
        _draw_labels(ax, grid, id2name, fontsize=label_fontsize)

    ax.set_title(title, fontsize=6)
    _axes_off(ax)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
