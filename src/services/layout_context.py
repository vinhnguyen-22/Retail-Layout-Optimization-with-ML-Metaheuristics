import ast
from typing import Dict, List, Optional, Tuple


class LayoutContext:
    """
    Gom các helper + cache liên quan đến layout:
    - seed layout từ layout thực
    - toạ độ slot + entrance
    - cat_support từ frequent itemsets
    - cắt layout theo số slot
    """

    def __init__(self, data, all_items: List[str], refrig_cats: List[str]):
        self.data = data
        self.all_items = all_items
        self.refrig_cats = refrig_cats
        self._baseline_cached: Optional[List[str]] = None
        self._coords_cached: Optional[
            Tuple[List[Tuple[float, float]], Tuple[float, float]]
        ] = None
        self._cat_support_cached: Optional[Dict[str, float]] = None

    def seed_layout_real(self) -> List[str]:
        if self._baseline_cached is not None:
            return self._baseline_cached
        known = set(self.all_items)
        baseline = (
            self.data.sorted_slots_xy()["Category"]
            .astype(str)
            .apply(lambda x: x if x in known else None)
            .dropna()
            .tolist()
        )
        self._baseline_cached = baseline
        return baseline

    def coords_and_entrance(
        self, override_entr_xy: Optional[Tuple[float, float]] = None
    ):
        if self._coords_cached is not None and override_entr_xy is None:
            return self._coords_cached

        slots = self.data.sorted_slots_xy().assign(
            width=lambda d: d["width"].fillna(0), height=lambda d: d["height"].fillna(0)
        )

        coords = list(
            zip(
                slots["x"] + slots["width"] * 0.5,
                slots["y"] + slots["height"] * 0.5,
            )
        )
        if override_entr_xy is not None:
            self._coords_cached = (coords, tuple(override_entr_xy))
            return self._coords_cached

        df = self.data.layout_real
        if ("is_entrance" in df.columns) and df["is_entrance"].fillna(0).astype(
            int
        ).any():
            row = (
                df.loc[df["is_entrance"].fillna(0).astype(int) == 1]
                .sort_values(["y", "x"])
                .iloc[0]
            )
        else:
            row = df.sort_values(["y", "x"]).iloc[0]

        ex = float(row["x"]) + float(row.get("width", 0)) * 0.5
        ey = float(row["y"]) + float(row.get("height", 0)) * 0.5
        self._coords_cached = (coords, (ex, ey))
        return self._coords_cached

    def cat_support(self) -> Dict[str, float]:
        if self._cat_support_cached is not None:
            return self._cat_support_cached
        cs = {c: 0.0 for c in self.all_items}
        df = self.data.freq_itemsets
        if "items" in df.columns and "support" in df.columns:
            for _, r in df.iterrows():
                try:
                    items = ast.literal_eval(r["items"])
                except Exception:
                    continue
                sup = float(r["support"])
                for it in items:
                    if it in cs:
                        cs[it] = max(cs[it], sup)
        self._cat_support_cached = cs
        return cs

    def trim_to_slots(self, layout: List[str]) -> List[str]:
        # dùng số slot thực có trong sorted_slots_xy
        return layout[: len(self.data.sorted_slots_xy())]
