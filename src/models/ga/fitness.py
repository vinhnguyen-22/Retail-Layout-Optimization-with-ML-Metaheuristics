import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def euclid(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


class FitnessComponents:
    def __init__(
        self,
        affinity_matrix: pd.DataFrame,
        coords: Optional[List[Tuple[float, float]]] = None,
        entr_xy: Optional[Tuple[float, float]] = None,
        cat_support: Optional[Dict[str, float]] = None,
        pairs_list: Optional[List[Tuple[str, str, float]]] = None,
        w_aff: float = 1.0,
        w_pair: float = 0.0,
        w_entr: float = 0.0,
        gamma_support: float = 0.7,
    ):
        self.aff = affinity_matrix
        self.coords = list(coords) if coords is not None else None
        self.entr_xy = entr_xy

        # 1) Gán trọng số trước
        self.w_aff = float(w_aff)
        self.w_pair = float(w_pair)
        self.w_entr = float(w_entr)
        self.gamma_support = float(gamma_support)

        # 2) Gán dữ liệu đầu vào (đổi sang container mặc định)
        self.pairs_list = list(pairs_list or [])
        self.cat_support = dict(cat_support or {})

        # 3) Short-circuit nếu weight = 0
        if self.w_pair == 0.0:
            self.pairs_list = []
        if self.w_entr == 0.0:
            self.cat_support = {}

        # 4) Luôn khởi tạo mean_dist mặc định
        self.mean_dist = 1.0
        if self.coords is not None and len(self.coords) >= 2:
            # tính nhanh bằng mẫu ngẫu nhiên (an toàn nếu coords ít phần tử)
            m = min(1000, len(self.coords) * 2)
            if m > 0:
                idx = np.random.randint(0, len(self.coords), size=(m, 2))
                d = [euclid(self.coords[i], self.coords[j]) for i, j in idx if i != j]
                if d:
                    self.mean_dist = float(np.mean(d))

    def score_affinity_adjacent(self, layout: List[str]) -> float:
        s = 0.0
        for i in range(len(layout) - 1):
            a, b = layout[i], layout[i + 1]
            try:
                s += float(self.aff.loc[a, b])
            except Exception:
                pass
        return s

    def cost_pairs_distance(self, layout: List[str]) -> float:
        """∑ w_ab * dist(slot(a), slot(b)) / mean_dist, CHỈ trên các mục có slot."""
        if not self.coords or not self.pairs_list:
            return 0.0

        # Chỉ map các category vào số slot sẵn có
        N = min(len(layout), len(self.coords))
        if N <= 1:
            return 0.0

        # Vị trí chỉ cho N phần tử đầu (có slot)
        pos = {layout[i]: i for i in range(N)}

        tot = 0.0
        for a, b, w in self.pairs_list:
            ia = pos.get(a)
            ib = pos.get(b)
            if ia is None or ib is None:
                continue
            # ia, ib chắc chắn < N nên không thể IndexError
            tot += float(w) * euclid(self.coords[ia], self.coords[ib])
        return tot / max(1e-9, self.mean_dist)

    def cost_entrance_distance(self, layout: List[str]) -> float:
        """∑ support(c)^gamma * dist(entr, slot(c)) / (mean_dist * N), CHỈ trên N mục có slot."""
        if not self.coords or self.entr_xy is None:
            return 0.0

        N = min(len(layout), len(self.coords))
        if N == 0:
            return 0.0

        tot = 0.0
        for i in range(N):
            c = layout[i]
            s = float(self.cat_support.get(c, 0.0)) ** self.gamma_support
            tot += s * euclid(self.entr_xy, self.coords[i])
        return tot / max(1e-9, self.mean_dist * N)

    def mixed_fitness(self, layout: List[str]) -> float:
        s = self.w_aff * self.score_affinity_adjacent(
            layout
        )  # điểm kề nhau theo affinity:contentReference[oaicite:0]{index=0}
        if self.w_pair > 0:
            s -= self.w_pair * self.cost_pairs_distance(
                layout
            )  # chỉ tính khi cần:contentReference[oaicite:1]{index=1}
        if self.w_entr > 0:
            s -= self.w_entr * self.cost_entrance_distance(
                layout
            )  # chỉ tính khi cần:contentReference[oaicite:2]{index=2}
        return s
