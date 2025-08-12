from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LayoutSpec:
    """Đặc tả layout + block tủ mát cố định (Two-Zone)."""

    all_items: List[str]
    refrig_cats: List[str]
    anchor_start: Optional[int] = None

    def __post_init__(self):
        self.all_items = list(self.all_items)
        self.set_all = set(self.all_items)
        self.refrig = [c for c in self.refrig_cats if c in self.set_all]
        self.set_refrig = set(self.refrig)
        self.N = len(self.all_items)
        self.R = len(self.refrig)
        if self.R == 0:
            raise ValueError("refrig_cats rỗng – Two-Zone cần block tủ mát.")

        if self.anchor_start is None:
            pos = [self.all_items.index(c) for c in self.refrig]
            self.anchor_start = min(pos)
        self.anchor_end = self.anchor_start + self.R
        if self.anchor_end > self.N:
            raise ValueError("Block tủ mát vượt quá độ dài layout.")
