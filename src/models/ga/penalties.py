from typing import Dict, Iterable, List, Optional, Tuple


class PenaltyRules:
    def __init__(
        self,
        forbidden_pairs: Iterable[Tuple[str, str]] = (),
        penalty_forbidden: float = 1000.0,
        hard_rules: Optional[
            Dict
        ] = None,  # {"must_together":[(a,b)], "must_order":[(a,b)]}
    ):
        self.forbidden = {(a, b) for a, b in forbidden_pairs} | {
            (b, a) for a, b in forbidden_pairs
        }
        self.penalty_forbidden = float(penalty_forbidden)
        self.hard_rules = hard_rules or {}

    def penalties(self, layout: List[str]) -> float:
        if not self.forbidden and not self.hard_rules:
            return 0.0
        pen = 0.0
        # Forbidden kề nhau
        for i in range(len(layout) - 1):
            a, b = layout[i], layout[i + 1]
            if (a, b) in self.forbidden:
                pen -= self.penalty_forbidden
        # Hard rules
        for a, b in self.hard_rules.get("must_together", []):
            if (
                a in layout
                and b in layout
                and abs(layout.index(a) - layout.index(b)) != 1
            ):
                pen -= 9_999
        for a, b in self.hard_rules.get("must_order", []):
            if a in layout and b in layout and layout.index(a) > layout.index(b):
                pen -= 9_999
        return pen
