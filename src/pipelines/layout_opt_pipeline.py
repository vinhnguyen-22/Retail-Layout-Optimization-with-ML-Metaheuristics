import random
from typing import List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from loguru import logger

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR
from src.models.affinity import AffinityBuilder
from src.models.ga_optimizer import GAOptimizer
from src.plots import LayoutVisualizer
from src.preprocess import DataLoader
from src.services.affinity_services import AffinityParams, AffinityService
from src.services.layout_context import LayoutContext
from src.services.layout_tuner import LayoutTuner


class LayoutOptimizationPipeline:
    """
    Pipeline tối ưu layout:
    - Tách Entrance/Cashier từ DataLoader (slots đã lọc sẵn).
    - Vector hoá tạo layout_opt, tránh copy không cần thiết.
    - Tuỳ chọn lọc category ‘lạc loài’ trước khi xuất/vẽ.
    """

    def __init__(
        self,
        data: DataLoader,
        n_trials: int = 30,
        n_gen_final: int = 80,
        use_optuna: bool = True,
        selection: str = "tournament",
        crossover: str = "PMX",
        mutation: str = "shuffle",
        adaptive: bool = True,
        seed: int = 42,
        # tuỳ chọn nâng cao
        pop_size: int = 200,
        elite_ratio: float = 0.06,
        write_logbook: bool = True,  # ghi logbook CSV
    ):
        self.data = data
        self.n_trials = n_trials
        self.n_gen_final = n_gen_final
        self.use_optuna = use_optuna
        self.selection, self.crossover, self.mutation = selection, crossover, mutation
        self.adaptive = adaptive
        self.seed = seed

        self.pop_size = pop_size
        self.elite_ratio = elite_ratio
        self.write_logbook = write_logbook

        np.random.seed(seed)
        random.seed(seed)

        self.all_items: List[str] = data.all_items
        self.refrig_cats: List[str] = data.refrig_cats

        self.aff_svc = AffinityService(
            AffinityBuilder(
                self.data.assoc_rules,
                self.data.freq_itemsets,
                self.data.all_items,
                self.data.margin_matrix,
            )
        )

        # Context
        self.ctx = LayoutContext(self.data, self.all_items, self.refrig_cats)

        # Tuner
        self.tuner = LayoutTuner(
            aff_svc=self.aff_svc,
            ctx=self.ctx,
            all_items=self.all_items,
            refrig_cats=self.refrig_cats,
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            adaptive=self.adaptive,
            seed=self.seed,
            n_trials=self.n_trials,
        )

        # holders
        self.best_params: Optional[dict] = None
        self.best_layout: Optional[List[str]] = None
        self.affinity = None
        self.layout_opt: Optional[pd.DataFrame] = None
        self.best_fitness: Optional[float] = None
        self.ga_logbook: Optional[pd.DataFrame] = None
        self.best_logbook: Optional[pd.DataFrame] = None

    # ---- Public API ----
    def tune(self):
        if not self.use_optuna:
            self.best_params = {
                "lift_threshold": 0.5,
                "w_lift": 0.6,
                "w_conf": 0.4,
                "w_margin": 0.0,
                "gamma": 1.0,
                "w_aff": 1.0,
                "w_entr": 0.0,
                "gamma_support": 0.0,
            }
            self.best_layout = self.ctx.trim_to_slots(self.ctx.seed_layout_real())
            logger.info("Tune skipped. Using default params.")
            return None

        best_params, best_layout, study = self.tuner.tune()
        self.best_params = best_params
        self.best_layout = best_layout

        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best layout (from Optuna): {self.best_layout}")
        return study

    def run_final(self) -> Tuple[pd.DataFrame, float]:
        if not self.best_params:
            raise RuntimeError("Hãy gọi tune() trước.")

        p = self.best_params
        self.affinity = self.aff_svc.build(
            AffinityParams(
                lift_threshold=p["lift_threshold"],
                w_lift=p["w_lift"],
                w_conf=p["w_conf"],
                w_margin=p["w_margin"],
                gamma=p["gamma"],
            )
        )

        # Tính coords/entr nếu cần
        coords = entr_xy = None
        cat_support = None
        if p["w_entr"] > 0:
            coords, entr_xy = self.ctx.coords_and_entrance()
            if p["gamma_support"] > 0:
                cat_support = self.ctx.cat_support()

        baseline = self.ctx.seed_layout_real()

        ga = GAOptimizer(
            all_items=self.all_items,
            affinity_matrix=self.affinity,
            refrig_cats=self.refrig_cats,
            hard_rules={},
            coords=coords,
            entr_xy=entr_xy,
            cat_support=cat_support,
            w_aff=p["w_aff"],
            w_entr=p["w_entr"],
            gamma_support=p["gamma_support"],
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
        )

        best_layout, best_fitness, logbook = ga.run(
            ngen=self.n_gen_final,
            pop_size=self.pop_size,
            seed=self.seed,
            elite_ratio=self.elite_ratio,
            adaptive=self.adaptive,
            baseline=baseline,
            as_dataframe=True,
        )

        # Lưu logbook (nếu chưa có)
        self.ga_logbook = getattr(ga, "logbook_df", None)
        if self.ga_logbook is None and isinstance(logbook, (list, pd.DataFrame)):
            try:
                self.ga_logbook = pd.DataFrame(logbook)
            except Exception:
                self.ga_logbook = None

        bl = list(map(str, self.ctx.trim_to_slots(self.best_layout or baseline)))
        # Lắp vào slots theo thứ tự (y,x) — vector hoá
        slots = self.data.sorted_slots_xy().iloc[: len(bl)].copy()
        slots.loc[:, "Category"] = pd.Series(bl, index=slots.index)
        slots["cx"] = slots["x"] + slots["width"] / 2.0
        slots["cy"] = slots["y"] + slots["height"] / 2.0

        self.layout_opt = slots[["Category", "x", "y", "width", "height", "cx", "cy"]]
        self.best_fitness = float(best_fitness)

        logger.info(f"Best fitness: {best_fitness:.4f}")
        return self.layout_opt, self.best_fitness

    def plot_all(self):
        if self.layout_opt is None or self.affinity is None:
            logger.info("Hãy chạy run_final() trước khi plot.")
            return

        # Affinity plots
        LayoutVisualizer.plot_affinity_heatmap(
            self.affinity, out_png=FIGURES_DIR / "affinity_heatmap.png"
        )
        LayoutVisualizer.plot_affinity_bar(
            self.affinity, out_png=FIGURES_DIR / "affinity_bar.png"
        )

        # GA curve (đổi tên không đè)
        log_df = self.ga_logbook if self.ga_logbook is not None else self.best_logbook
        if log_df is not None and not log_df.empty:
            LayoutVisualizer.plot_ga_convergence(
                log_df, out_png=FIGURES_DIR / "ga_convergence.png"
            )

        # Spring layout + visualize
        LayoutVisualizer.plot_spring_layout(
            self.affinity, threshold=0.8, out_png=FIGURES_DIR / "spring_layout.png"
        )
        LayoutVisualizer.plot_visualize_layout(
            df_layout=self.layout_opt,
            out_png=FIGURES_DIR / "ga_preview.png",
            show_labels=True,
        )
        LayoutVisualizer.plot_compare_layouts(
            df_after=self.layout_opt,
            df_before=self.data.layout_real,
            out_png=FIGURES_DIR / "ga_compare.png",
        )
