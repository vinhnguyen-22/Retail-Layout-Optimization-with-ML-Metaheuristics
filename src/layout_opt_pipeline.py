# src/pipelines/pipeline_layout_opt.py
import random
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
import typer
from loguru import logger

from src.config import INTERIM_DATA_DIR, OUTPUT_DATA_DIR, PROCESSED_DATA_DIR
from src.models.affinity import AffinityBuilder
from src.models.ga_optimizer import GAOptimizer
from src.models.greedy import GreedyLayout
from src.plots import LayoutVisualizer
from src.preprocess import DataLoader
from src.services.affinity_services import AffinityParams, AffinityService
from src.services.layout_context import LayoutContext
from src.services.layout_tuner import LayoutTuner

app = typer.Typer(help="Retail Forecast Pipeline CLI")


# =============== Pipeline ===============
class LayoutOptimizationPipeline:
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
    ):
        self.data = data
        self.n_trials = n_trials
        self.n_gen_final = n_gen_final
        self.use_optuna = use_optuna
        self.selection, self.crossover, self.mutation = selection, crossover, mutation
        self.adaptive = adaptive
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        self.all_items: List[str] = data.all_items
        self.refrig_cats: List[str] = data.refrig_cats

        # Affinity builder gốc
        self.affinity_builder = AffinityBuilder(
            self.data.assoc_rules,
            self.data.freq_itemsets,
            self.data.all_items,
            self.data.margin_matrix,
        )

        # Services
        self.aff_svc = AffinityService(self.affinity_builder)
        self.ctx = LayoutContext(self.data, self.all_items, self.refrig_cats)

        # Tuner (Optuna)
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
        self.study: Optional[optuna.Study] = None
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
            # chạy nhanh không tune: giữ tham số mặc định hợp lý
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
            self.study = None
            self.best_logbook = None
            logger.info("Tune skipped. Using default params.")
            return None

        best_params, best_layout, study = self.tuner.tune()
        self.study = study
        self.best_params = best_params
        self.best_layout = best_layout
        self.best_logbook = None

        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best layout (from Optuna): {self.best_layout}")
        return study

    def run_final(self):
        if not hasattr(self, "best_params") or self.best_params is None:
            raise RuntimeError("Hãy gọi tune() trước.")

        p = self.best_params
        affinity = self.aff_svc.build(
            AffinityParams(
                lift_threshold=p["lift_threshold"],
                w_lift=p["w_lift"],
                w_conf=p["w_conf"],
                w_margin=p["w_margin"],
                gamma=p["gamma"],
            )
        )

        # lazy theo w_entr/gamma_support
        coords = entr_xy = None
        cat_support = None
        if p["w_entr"] > 0:
            coords, entr_xy = self.ctx.coords_and_entrance()
            if p["gamma_support"] > 0:
                cat_support = self.ctx.cat_support()

        baseline = self.ctx.seed_layout_real()

        # One-optimizer
        ga = GAOptimizer(
            all_items=self.all_items,
            affinity_matrix=affinity,
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

        # lưu logbook final ra CSV
        log_csv = PROCESSED_DATA_DIR / "ga_logbook_final.csv"

        best_layout, best_fitness, logbook = ga.run(
            ngen=self.n_gen_final,
            pop_size=200,  # cố định để repeatable
            seed=self.seed,
            elite_ratio=0.06,  # cố định
            adaptive=self.adaptive,
            baseline=baseline,
            log_csv_path=str(log_csv),
            as_dataframe=True,
        )

        # lưu logbook để vẽ
        self.ga_logbook = getattr(ga, "logbook_df", None)
        if self.ga_logbook is None:
            try:
                self.ga_logbook = pd.DataFrame(logbook)
            except Exception:
                self.ga_logbook = None

        # Xuất file theo slot (y,x)
        best_layout = [str(c) for c in self.ctx.trim_to_slots(best_layout)]
        slots = self.data.sorted_slots_xy()
        n = min(len(best_layout), len(slots))
        layout_opt = pd.DataFrame(
            {
                "Category": best_layout[:n],
                "x": slots.loc[: n - 1, "x"].to_list(),
                "y": slots.loc[: n - 1, "y"].to_list(),
                "width": slots.loc[: n - 1, "width"].to_list(),
                "height": slots.loc[: n - 1, "height"].to_list(),
            }
        )
        layout_opt["cx"] = layout_opt["x"] + layout_opt["width"] / 2.0
        layout_opt["cy"] = layout_opt["y"] + layout_opt["height"] / 2.0

        self.affinity = affinity
        self.layout_opt = layout_opt
        self.best_fitness = best_fitness

        logger.info(f"\nBest layout: {best_layout}")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        return layout_opt, best_fitness

    def plot_all(self):
        if not hasattr(self, "layout_opt") or self.layout_opt is None:
            logger.info("Hãy chạy run_final() trước khi plot.")
            return
        LayoutVisualizer.plot_affinity_heatmap(self.affinity)
        LayoutVisualizer.plot_affinity_bar(self.affinity)
        # Ưu tiên logbook final; nếu không có thì dùng best_logbook từ tune()
        log_df = self.ga_logbook if self.ga_logbook is not None else self.best_logbook
        if log_df is not None and not log_df.empty:
            LayoutVisualizer.plot_ga_convergence(log_df)
        LayoutVisualizer.plot_spring_layout(self.affinity, threshold=0.8)
        LayoutVisualizer.plot_visualize_layout(
            self.layout_opt,
            out_png=OUTPUT_DATA_DIR / "ga_preview.png",
            show_labels=True,
        )
        LayoutVisualizer.plot_compare_layouts(
            df_after=self.layout_opt,
            df_before=self.data.layout_real,
            out_png=OUTPUT_DATA_DIR / "ga_compare.png",
            show_labels=True,
        )


# =============== Example usage ===============
@app.command("run")
def run(
    assoc_rules_path: str = str("association_rules.csv"),
    freq_itemsets_path: str = str("frequent_itemsets.csv"),
    layout_real_path: str = str("layout.csv"),
    margin_matrix_path: str = None,
    n_trials: int = 20,
    n_gen_final: int = 80,
    selection: str = "tournament",
    crossover: str = "PMX",
    mutation: str = "shuffle",
    adaptive: bool = True,
    seed: int = 42,
):
    data = DataLoader(
        assoc_rules_path=PROCESSED_DATA_DIR / assoc_rules_path,
        freq_itemsets_path=PROCESSED_DATA_DIR / freq_itemsets_path,
        layout_real_path=INTERIM_DATA_DIR / layout_real_path,
        margin_matrix_path=margin_matrix_path,
    )

    pipeline = LayoutOptimizationPipeline(
        data=data,
        n_trials=n_trials,
        n_gen_final=n_gen_final,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        adaptive=adaptive,
        seed=seed,
    )

    pipeline.tune()
    pipeline.run_final()
    pipeline.plot_all()


if __name__ == "__main__":
    app()
