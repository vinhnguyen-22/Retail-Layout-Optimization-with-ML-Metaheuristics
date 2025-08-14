from typing import Dict, Optional

import optuna

from src.models.ga_optimizer import GAOptimizer
from src.models.greedy import GreedyLayout
from src.services.affinity_services import AffinityParams, AffinityService
from src.services.layout_context import LayoutContext


class LayoutTuner:
    """
    Gói toàn bộ quá trình tune bằng Optuna:
    - Sinh tham số (affinity + fitness)
    - Tạo affinity qua AffinityService
    - Lấy seed/coords/cat_support qua LayoutContext
    - Chạy GA 30 thế hệ để đánh giá
    Trả ra best_params + best_layout (cắt theo số slot).
    """

    def __init__(
        self,
        *,
        aff_svc: AffinityService,
        ctx: LayoutContext,
        all_items,
        refrig_cats,
        selection: str,
        crossover: str,
        mutation: str,
        adaptive: bool,
        seed: int,
        n_trials: int = 30,
    ):
        self.aff_svc = aff_svc
        self.ctx = ctx
        self.all_items = all_items
        self.refrig_cats = refrig_cats
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.adaptive = adaptive
        self.seed = seed
        self.n_trials = n_trials

        # output
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None
        self.best_layout = None

    def _objective(self, trial: optuna.trial.Trial) -> float:
        # ---- sample params ----
        lift_threshold = trial.suggest_float("lift_threshold", 0.0, 2.0)
        w_lift = trial.suggest_float("w_lift", 0.1, 1.5)
        w_conf = trial.suggest_float("w_conf", 0.0, 1.2)
        w_margin = trial.suggest_float("w_margin", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.3, 5.0, log=True)

        w_aff = trial.suggest_float("w_aff", 0.4, 2.0)
        w_entr = trial.suggest_float("w_entr", 0.0, 1.0)
        gamma_support = trial.suggest_float("gamma_support", 1e-3, 1.5, log=True)

        # diện tích
        w_area = trial.suggest_float("w_area", 0.3, 1.5)
        w_area_slack = trial.suggest_float("w_area_slack", 0.0, min(0.5, 0.5 * w_area))

        # --- NEW: sector weights ---
        w_sector_adj = trial.suggest_float("w_sector_adj", 0.0, 0.6)
        w_sector_disp = trial.suggest_float("w_sector_disp", 0.0, 1)

        # ---- build affinity ----
        affinity = self.aff_svc.build(
            AffinityParams(
                lift_threshold=lift_threshold,
                w_lift=w_lift,
                w_conf=w_conf,
                w_margin=w_margin,
                gamma=gamma,
            )
        )

        # ---- seed & contexts ----
        baseline = self.ctx.seed_layout_real()
        layout_greedy = None
        if not baseline:
            greedy = GreedyLayout(self.all_items)
            layout_greedy = greedy.local_search(greedy.init_layout(affinity), affinity)

        coords = entr_xy = None
        cat_support = None
        if w_entr > 0:
            coords, entr_xy = self.ctx.coords_and_entrance()
            if gamma_support > 0:
                cat_support = self.ctx.cat_support()

        seed_layout = baseline or layout_greedy

        # diện tích
        n_slots = len(seed_layout)
        data_obj = getattr(self.ctx, "data", None)
        slot_area = (
            data_obj.get_slot_area(n_slots)
            if (data_obj and hasattr(data_obj, "get_slot_area"))
            else [1.0] * n_slots
        )
        item_area = (
            data_obj.get_item_area_dict(self.all_items, default=1.0)
            if (data_obj and hasattr(data_obj, "get_item_area_dict"))
            else {cat: 1.0 for cat in self.all_items}
        )

        # --- NEW: sector map (an toàn nếu thiếu) ---
        category_sector = getattr(data_obj, "category_sector", {}) if data_obj else {}

        # ---- GA ----
        ga = GAOptimizer(
            all_items=self.all_items,
            affinity_matrix=affinity,
            refrig_cats=self.refrig_cats,
            hard_rules={},
            coords=coords,
            entr_xy=entr_xy,
            cat_support=cat_support,
            # weights hiện có
            w_aff=w_aff,
            w_entr=w_entr,
            gamma_support=gamma_support,
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            anchor_start=None,
            # diện tích
            item_area=item_area,
            slot_area=slot_area,
            w_area=w_area,
            w_area_slack=w_area_slack,
            # --- NEW: sector ---
            category_sector=category_sector,
            w_sector_adj=w_sector_adj,
            w_sector_disp=w_sector_disp,
        )

        best_layout, best_fitness, _ = ga.run(
            ngen=30,
            pop_size=200,
            seed=self.seed,
            elite_ratio=0.06,
            adaptive=self.adaptive,
            baseline=seed_layout,
            log_csv_path=None,
            as_dataframe=True,
        )

        trial.set_user_attr("best_layout", best_layout)
        return float(best_fitness)

    def tune(self):
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self._objective, n_trials=self.n_trials)
        self.study = study
        self.best_params = study.best_params
        self.best_layout = study.best_trial.user_attrs["best_layout"]
        return self.best_params, self.best_layout, study
