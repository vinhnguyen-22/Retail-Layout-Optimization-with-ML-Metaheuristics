from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.models.ga.decoder import DecoderTwoZone, IdentityDecoder
from src.models.ga.fitness import FitnessComponents
from src.models.ga.penalties import PenaltyRules
from src.models.ga.permuation import PermutationGA
from src.validations.spec import LayoutSpec


class GAOptimizer:
    def __init__(
        self,
        all_items: List[str],
        affinity_matrix: pd.DataFrame,
        refrig_cats: Optional[List[str]] = None,
        hard_rules: Optional[Dict] = None,
        coords: Optional[List[Tuple[float, float]]] = None,
        entr_xy: Optional[Tuple[float, float]] = None,
        cat_support: Optional[Dict[str, float]] = None,
        w_aff: float = 1.0,
        w_entr: float = 0.0,
        gamma_support: float = 0.7,
        selection: str = "tournament",
        crossover: str = "PMX",
        mutation: str = "shuffle",
        anchor_start: Optional[int] = None,
    ):
        # fitness/penalties
        fitc = FitnessComponents(
            affinity_matrix=affinity_matrix,
            coords=coords,
            entr_xy=entr_xy,
            cat_support=cat_support,
            w_aff=w_aff,
            w_pair=0.0,  # đã bỏ pairs
            w_entr=w_entr,
            gamma_support=gamma_support,
        )
        pens = PenaltyRules(
            forbidden_pairs=(),  # đã bỏ cấm động theo affinity==0
            penalty_forbidden=0.0,
            hard_rules=hard_rules or {},
        )

        # decoder: Two-Zone nếu có refrig_cats hợp lệ, ngược lại identity
        refrig_cats = refrig_cats or []
        if len(refrig_cats) > 0:
            spec = LayoutSpec(
                all_items=all_items, refrig_cats=refrig_cats, anchor_start=anchor_start
            )
            decoder = DecoderTwoZone(spec)
        else:
            decoder = IdentityDecoder()

        # GA engine chung
        self.ga = PermutationGA(
            items=list(all_items),
            decoder=decoder,
            fitc=fitc,
            pens=pens,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
        )
        self.logbook = None
        self.logbook_df = None

    def run(
        self,
        ngen: int = 60,
        pop_size: int = 200,
        seed: Optional[int] = None,
        elite_ratio: float = 0.06,
        adaptive: bool = True,
        baseline: Optional[List[str]] = None,
        log_csv_path: Optional[str] = None,
        as_dataframe: bool = True,
    ):
        best_layout, best_fitness, logbook = self.ga.run(
            ngen=ngen,
            pop_size=pop_size,
            seed=seed,
            elite_ratio=elite_ratio,
            adaptive=adaptive,
            baseline=baseline,
        )
        self.logbook = logbook
        if as_dataframe:
            try:
                self.logbook_df = pd.DataFrame(logbook)
                if log_csv_path:
                    self.logbook_df.to_csv(log_csv_path, index=False)
            except Exception:
                self.logbook_df = None

        return best_layout, best_fitness, logbook
