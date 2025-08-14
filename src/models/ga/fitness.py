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
        # --- NEW: diện tích ---
        item_area: Optional[Dict[str, float]] = None,  # diện tích/footprint mỗi item
        slot_area: Optional[
            List[float]
        ] = None,  # sức chứa diện tích mỗi slot (align với coords[i])
        # --- NEW: sector ---
        category_sector: Optional[Dict[str, str]] = None,  # map Category -> SectorCode
        # weights
        w_aff: float = 1.0,
        w_pair: float = 0.0,
        w_entr: float = 0.0,
        w_area: float = 0.0,  # phạt overflow (vượt sức chứa)
        w_area_slack: float = 0.0,  # (tùy chọn) phạt phần thừa nhẹ
        # --- NEW: sector weights ---
        w_sector_adj: float = 0.0,  # thưởng kề nhau cùng sector
        w_sector_disp: float = 0.0,  # phạt phân tán sector
        gamma_support: float = 0.7,
    ):
        self.aff = affinity_matrix
        self.coords = list(coords) if coords is not None else None
        self.entr_xy = entr_xy

        # weights
        self.w_aff = float(w_aff)
        self.w_pair = float(w_pair)
        self.w_entr = float(w_entr)
        self.w_area = float(w_area)
        self.w_area_slack = float(w_area_slack)
        self.w_sector_adj = float(w_sector_adj)
        self.w_sector_disp = float(w_sector_disp)
        self.gamma_support = float(gamma_support)

        # inputs
        self.pairs_list = list(pairs_list or [])
        self.cat_support = dict(cat_support or {})

        # area inputs
        self.item_area = dict(item_area or {})
        self.slot_area = list(slot_area or [])

        # sector input
        self.category_sector = dict(category_sector or {})

        # short-circuit để tiết kiệm nếu không dùng
        if self.w_pair == 0.0:
            self.pairs_list = []
        if self.w_entr == 0.0:
            self.cat_support = {}
        if (self.w_area == 0.0) and (self.w_area_slack == 0.0):
            self.item_area = {}
            self.slot_area = []
        if (self.w_sector_adj == 0.0) and (self.w_sector_disp == 0.0):
            self.category_sector = {}

        # mean distance để chuẩn hóa các chi phí liên quan khoảng cách
        self.mean_dist = 1.0
        if self.coords is not None and len(self.coords) >= 2:
            m = min(1000, len(self.coords) * 2)
            if m > 0:
                idx = np.random.randint(0, len(self.coords), size=(m, 2))
                d = [euclid(self.coords[i], self.coords[j]) for i, j in idx if i != j]
                if d:
                    self.mean_dist = float(np.mean(d))

    # ---------- thành phần điểm/chi phí hiện có ----------
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
        N = min(len(layout), len(self.coords))
        if N <= 1:
            return 0.0
        pos = {layout[i]: i for i in range(N)}
        tot = 0.0
        for a, b, w in self.pairs_list:
            ia = pos.get(a)
            ib = pos.get(b)
            if ia is None or ib is None:
                continue
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

    # ---------- NEW: sector ----------
    def score_sector_adjacent(self, layout: List[str]) -> float:
        """
        Thưởng khi 2 slot kề nhau thuộc cùng sector.
        Trả về tỉ lệ cặp kề 'match sector' trên (N-1) để chuẩn hoá [0,1].
        """
        if not self.category_sector:
            return 0.0
        N = len(layout)
        if N <= 1:
            return 0.0
        hit = 0
        for i in range(N - 1):
            a, b = layout[i], layout[i + 1]
            if self.category_sector.get(a) and self.category_sector.get(
                a
            ) == self.category_sector.get(b):
                hit += 1
        return hit / (N - 1)

    def cost_sector_dispersion(self, layout: List[str]) -> float:
        """
        Phạt sự phân tán của mỗi sector.
        - Nếu có coords: tính 'tổng khoảng cách tới centroid' cho từng sector,
          chuẩn hoá theo mean_dist và số phần tử sector.
        - Nếu không có coords: dùng chỉ số slot (i) như vị trí 1D và khoảng cách |i - mean_i|.
        Trả về giá trị >= 0, đã chuẩn hoá theo tổng số item.
        """
        if not self.category_sector:
            return 0.0

        N_eff = self._effective_N(layout)
        if N_eff == 0:
            return 0.0

        # gom index các item theo sector (chỉ xét trong phần có slot/coords)
        sector_to_idx: Dict[str, List[int]] = {}
        for i in range(N_eff):
            c = layout[i]
            s = self.category_sector.get(c)
            if not s:
                continue
            sector_to_idx.setdefault(s, []).append(i)

        if not sector_to_idx:
            return 0.0

        total = 0.0
        count_items = 0

        if self.coords is not None and len(self.coords) >= N_eff:
            # dùng toạ độ 2D
            for s, idxs in sector_to_idx.items():
                if len(idxs) <= 1:
                    count_items += len(idxs)
                    continue
                pts = np.array([self.coords[i] for i in idxs], dtype=float)
                centroid = pts.mean(axis=0)
                dists = np.linalg.norm(pts - centroid, axis=1)  # L2
                # chuẩn hoá theo mean_dist để scale ~ O(1)
                total += float(np.sum(dists)) / max(1e-9, self.mean_dist)
                count_items += len(idxs)
        else:
            # fallback 1D theo chỉ số slot
            for s, idxs in sector_to_idx.items():
                if len(idxs) <= 1:
                    count_items += len(idxs)
                    continue
                arr = np.array(idxs, dtype=float)
                mean_i = arr.mean()
                dists = np.abs(arr - mean_i)
                # chuẩn hoá theo N_eff để giữ scale hợp lý
                total += float(np.sum(dists)) / max(1e-9, N_eff)
                count_items += len(idxs)

        # chuẩn hoá theo tổng số item đã xét (trung bình per-item)
        return total / max(1e-9, count_items)

    # ---------- NEW: chi phí diện tích ----------
    def _effective_N(self, layout: List[str]) -> int:
        """Số vị trí thực sự có thể đánh giá (layout, slot_area, và coords nếu có)."""
        N_candidates = [len(layout)]
        if self.slot_area:
            N_candidates.append(len(self.slot_area))
        if self.coords is not None:
            N_candidates.append(len(self.coords))
        return min(N_candidates) if N_candidates else 0

    def cost_area_overflow(self, layout: List[str]) -> float:
        """
        Soft-penalty overflow:
          overflow_ratio = sum(max(0, area(item) - cap(slot_i))) / sum(area(item))
        """
        if not self.item_area or not self.slot_area:
            return 0.0
        N = self._effective_N(layout)
        if N == 0:
            return 0.0
        sum_overflow = 0.0
        sum_item_area = 0.0
        for i in range(N):
            a = float(self.item_area.get(layout[i], 0.0))
            cap = float(self.slot_area[i])
            overflow = max(0.0, a - max(0.0, cap))
            sum_overflow += overflow
            sum_item_area += max(0.0, a)
        if sum_item_area <= 0.0:
            denom = max(1e-9, float(np.sum(self.slot_area[:N])))
        else:
            denom = sum_item_area
        return sum_overflow / max(1e-9, denom)

    def cost_area_slack(self, layout: List[str], delta_ratio: float = 0.2) -> float:
        """
        Phạt nhẹ phần 'thừa' (slack = cap - area) theo Huber-like:
          - Quadratic khi slack nhỏ (<= delta), tuyến tính khi lớn.
        delta ~ tỉ lệ theo area(item): delta = delta_ratio * area(item)
        """
        if not self.item_area or not self.slot_area:
            return 0.0
        N = self._effective_N(layout)
        if N == 0:
            return 0.0
        slack_sum = 0.0
        area_sum = 0.0
        for i in range(N):
            a = float(self.item_area.get(layout[i], 0.0))
            cap = float(self.slot_area[i])
            slack = max(0.0, max(0.0, cap) - max(0.0, a))
            # ngưỡng “nhỏ” theo tỷ lệ diện tích item (fallback 1.0 nếu a=0)
            delta = delta_ratio * (a if a > 0 else 1.0)
            if slack <= delta:
                slack_sum += (slack**2) / (2 * delta + 1e-9)
            else:
                slack_sum += slack - delta / 2.0
            area_sum += max(0.0, a)
        denom = (
            area_sum if area_sum > 0 else max(1e-9, float(np.sum(self.slot_area[:N])))
        )
        return slack_sum / max(1e-9, denom)

    # ---------- tổng hợp ----------
    def mixed_fitness(self, layout: List[str]) -> float:
        s = self.w_aff * self.score_affinity_adjacent(layout)
        if self.w_pair > 0:
            s -= self.w_pair * self.cost_pairs_distance(layout)
        if self.w_entr > 0:
            s -= self.w_entr * self.cost_entrance_distance(layout)
        if self.w_area > 0:
            s -= self.w_area * self.cost_area_overflow(layout)
        if self.w_area_slack > 0:
            s -= self.w_area_slack * self.cost_area_slack(layout)
        if self.w_sector_adj > 0:
            s += self.w_sector_adj * self.score_sector_adjacent(layout)
        if self.w_sector_disp > 0:
            s -= self.w_sector_disp * self.cost_sector_dispersion(layout)
        return s
