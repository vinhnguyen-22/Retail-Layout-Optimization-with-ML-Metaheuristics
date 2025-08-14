import random
from typing import Dict, List, Optional, Tuple

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
    - ĐÃ THÊM: hỗ trợ diện tích (item_area, slot_area) cho GA.
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

    # ---- NEW: helper diện tích ----
    def _areas_from_data(self, n_slots: int) -> Tuple[Dict[str, float], List[float]]:
        """
        Trả về (item_area, slot_area) khớp với layout/coords.
        - slot_area: lấy từ sorted_slots_xy() -> width * height cho từng slot theo thứ tự.
        - item_area: ưu tiên từ self.data (nếu có), bằng 1.0 nếu không có dữ liệu.
        """
        slots_df = self.data.sorted_slots_xy().iloc[:n_slots].copy()
        # đảm bảo có cột width/height
        if not {"width", "height"}.issubset(slots_df.columns):
            # fallback: nếu không có, đặt 1.0 cho mọi slot
            slot_area = [1.0] * len(slots_df)
        else:
            slot_area = (slots_df["width"] * slots_df["height"]).astype(float).tolist()

        # 2) item_area theo category
        item_area: Dict[str, float] = {}

        # ưu tiên: DataLoader cung cấp sẵn (nếu bạn có thuộc tính này)
        if hasattr(self.data, "item_area") and isinstance(self.data.item_area, dict):
            # kỳ vọng keys là Category
            for cat in self.all_items:
                item_area[cat] = float(self.data.item_area.get(cat, 1.0))

        else:
            # không có dữ liệu → gán 1.0 (chuẩn hóa tương đối)
            item_area = {cat: 1.0 for cat in self.all_items}

        return item_area, slot_area

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
                # --- diện tích ---
                "w_area": 0.6,
                "w_area_slack": 0.1,
                # --- NEW: sector (default) ---
                "w_sector_adj": 0.2,
                "w_sector_disp": 0.5,
            }
            self.best_layout = self.ctx.seed_layout_real()
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

        # Tính coords/entr nếu cần cho entrance
        coords = entr_xy = None
        cat_support = None
        if p["w_entr"] > 0:
            coords, entr_xy = self.ctx.coords_and_entrance()
            if p["gamma_support"] > 0:
                cat_support = self.ctx.cat_support()

        baseline = self.ctx.seed_layout_real()
        n_slots = len(baseline)

        # Diện tích & sector
        item_area = self.data.get_item_area_dict(self.all_items, default=1.0)
        slot_area = self.data.get_slot_area(n_slots)
        category_sector = getattr(self.data, "category_sector", {})

        ga = GAOptimizer(
            all_items=self.all_items,
            affinity_matrix=self.affinity,
            refrig_cats=self.refrig_cats,
            hard_rules={},
            coords=coords,
            entr_xy=entr_xy,
            cat_support=cat_support,
            # weights
            w_aff=p["w_aff"],
            w_entr=p["w_entr"],
            gamma_support=p["gamma_support"],
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            # diện tích
            item_area=item_area,
            slot_area=slot_area,
            w_area=p.get("w_area", 0.6),
            w_area_slack=p.get("w_area_slack", 0.1),
            # sector
            category_sector=category_sector,
            w_sector_adj=p.get("w_sector_adj", 0.2),
            w_sector_disp=p.get("w_sector_disp", 0.5),
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

        # Lưu logbook nếu có
        self.ga_logbook = getattr(ga, "logbook_df", None)
        if self.ga_logbook is None and isinstance(logbook, (list, pd.DataFrame)):
            try:
                self.ga_logbook = pd.DataFrame(logbook)
            except Exception:
                self.ga_logbook = None

        # Build output theo thứ tự (y,x)
        slots = self.data.sorted_slots_xy().copy()
        slot_cats = slots["Category"].astype(str).tolist()

        # Nếu GA đã trả đúng số slot thì dùng trực tiếp
        if len(best_layout) == len(slots):
            bl = list(map(str, best_layout))
        else:
            # GA trả thứ tự category duy nhất -> nở theo bội số có sẵn trong layout thực tế
            order = list(map(str, best_layout))
            mult = pd.Series(slot_cats).value_counts()
            remaining: Dict[str, int] = mult.to_dict()
            expanded: List[str] = []

            # 1) Đặt các block theo thứ tự GA
            for cat in order:
                k = int(remaining.pop(cat, 0))
                if k > 0:
                    expanded.extend([cat] * k)

            # 2) Nếu còn category chưa xếp (GA không nhắc đến), thêm theo thứ tự gốc của slots
            if len(expanded) < len(slots):
                for cat in slot_cats:
                    if remaining.get(cat, 0) > 0:
                        expanded.append(cat)
                        remaining[cat] -= 1
                        if remaining[cat] == 0:
                            remaining.pop(cat, None)
                    if len(expanded) == len(slots):
                        break

            bl = expanded

        # Safety: độ dài phải khớp số slot
        assert len(bl) == len(slots), f"len(bl)={len(bl)} != len(slots)={len(slots)}"

        # Gán & tính thêm toạ độ center
        slots["Category"] = bl
        slots["cx"] = slots["x"] + slots["width"] / 2.0
        slots["cy"] = slots["y"] + slots["height"] / 2.0

        # Lưu state
        self.best_layout = bl  # slot-level
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
        LayoutVisualizer.plot_visualize_layout_plotly(
            df_layout=self.layout_opt,
            out_html=FIGURES_DIR / "ga_preview.html",
            show_labels=True,  # hoặc False nếu rối mắt
            label_fontsize=8,
            title="Layout (GA) — interactive",
        )

        LayoutVisualizer.plot_compare_layouts(
            df_after=self.layout_opt,
            df_before=self.data.layout_real,
            out_png=FIGURES_DIR / "ga_compare.png",
        )
