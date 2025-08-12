from dataclasses import dataclass


@dataclass
class AffinityParams:
    lift_threshold: float = 0.5
    w_lift: float = 0.6
    w_conf: float = 0.4
    w_margin: float = 0.0
    gamma: float = 1.0


class AffinityService:
    def __init__(self, builder):
        self.builder = builder

    def build(self, p: AffinityParams):
        # Chuẩn hoá trọng số để tổng = 1
        s = p.w_lift + p.w_conf + p.w_margin
        if s <= 0:
            s = 1.0
        aff = self.builder.build_affinity(
            lift_threshold=p.lift_threshold,
            w_lift=p.w_lift / s,
            w_conf=p.w_conf / s,
            w_margin=p.w_margin / s,
        )
        aff = self.builder.normalize(aff)
        return self.builder.kernelize(aff, gamma=p.gamma)
