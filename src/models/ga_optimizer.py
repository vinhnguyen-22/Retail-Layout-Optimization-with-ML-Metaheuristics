# src/models/layout_ga_engines.py
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools


# =========================
# Utils
# =========================
def euclid(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# =========================
# Spec & Decoder (Two-Zone)
# =========================
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


class DecoderTwoZone:
    """Giải mã genome (hoán vị toàn cục) -> layout 2 vùng (block tủ mát cố định)."""

    def __init__(self, spec: LayoutSpec):
        self.spec = spec

    def decode(self, genome: List[str]) -> List[str]:
        if set(genome) != self.spec.set_all or len(genome) != self.spec.N:
            raise ValueError("Genome không phải hoán vị đúng của all_items.")
        start, end = self.spec.anchor_start, self.spec.anchor_end
        prefix_len = start
        suffix_len = self.spec.N - end

        non_refrig_seq = [g for g in genome if g not in self.spec.set_refrig]
        refrig_seq = [g for g in genome if g in self.spec.set_refrig]

        prefix = non_refrig_seq[:prefix_len]
        suffix = non_refrig_seq[prefix_len : prefix_len + suffix_len]
        return prefix + refrig_seq + suffix


# =========================
# Fitness & Penalties (dùng chung)
# =========================
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
        self.cat_support = cat_support or {}
        self.pairs_list = pairs_list or []
        self.w_aff, self.w_pair, self.w_entr = (
            float(w_aff),
            float(w_pair),
            float(w_entr),
        )
        self.gamma_support = float(gamma_support)

        if self.coords is not None and len(self.coords) >= 2:
            m = min(1000, len(self.coords) * 2)
            idx = np.random.randint(0, len(self.coords), size=(m, 2))
            d = [euclid(self.coords[i], self.coords[j]) for i, j in idx if i != j]
            self.mean_dist = float(np.mean(d)) if d else 1.0
        else:
            self.mean_dist = 1.0

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
        return (
            self.w_aff * self.score_affinity_adjacent(layout)
            - self.w_pair * self.cost_pairs_distance(layout)
            - self.w_entr * self.cost_entrance_distance(layout)
        )


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


# =========================
# GA base helpers
# =========================
def _safe_define_deap():
    for cname in ["FitnessMax", "Individual"]:
        if hasattr(creator, cname):
            delattr(creator, cname)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


class _BaseGA:
    @staticmethod
    def diversity(pop):
        if not pop:
            return 0.0
        return len({tuple(ind) for ind in pop}) / len(pop)

    @staticmethod
    def _index_op_categories(ind1, ind2, items, op):
        cat2i = {c: i for i, c in enumerate(items)}
        a1 = [cat2i[c] for c in ind1]
        a2 = [cat2i[c] for c in ind2]
        op(a1, a2)
        i2cat = {i: c for c, i in cat2i.items()}
        ind1[:] = [i2cat[i] for i in a1]
        ind2[:] = [i2cat[i] for i in a2]
        return ind1, ind2

    @staticmethod
    def _index_mut_categories(ind, items, op, **kwargs):
        cat2i = {c: i for i, c in enumerate(items)}
        arr = [cat2i[c] for c in ind]
        op(arr, **kwargs)
        i2cat = {i: c for c, i in cat2i.items()}
        ind[:] = [i2cat[i] for i in arr]
        return (ind,)


# NEW: helper hoàn thiện hoán vị từ seed (vá baseline ngắn/thừa)
def _complete_perm_from_seed(all_items: List[str], seed_order: List[str]) -> List[str]:
    """Giữ thứ tự seed hợp lệ, thêm phần còn thiếu theo thứ tự all_items."""
    seen = set()
    seed_clean = []
    set_all = set(all_items)
    for c in seed_order or []:
        if c in set_all and c not in seen:
            seed_clean.append(c)
            seen.add(c)
    rest = [c for c in all_items if c not in seen]
    return seed_clean + rest


# =========================
# Two-Zone GA
# =========================
class TwoZonePermutationGA(_BaseGA):
    """GA trên hoán vị toàn cục; decode 2-vùng (block tủ mát cố định)."""

    def __init__(
        self,
        spec: LayoutSpec,
        decoder: DecoderTwoZone,
        fitc: FitnessComponents,
        pens: PenaltyRules,
        selection: str = "tournament",
        crossover: str = "PMX",
        mutation: str = "shuffle",
    ):
        _safe_define_deap()
        self.spec, self.decoder, self.fitc, self.pens = spec, decoder, fitc, pens

        self.items = list(spec.all_items)
        self.toolbox = base.Toolbox()
        self._register_ops(selection, crossover, mutation)

    def _register_ops(self, selection: str, crossover: str, mutation: str):
        items = self.items

        def _rand_perm():
            g = items[:]
            random.shuffle(g)
            return g

        self.toolbox.register("indices", _rand_perm)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.indices
        )
        # FIX: dùng initRepeat để hỗ trợ population(n=...)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        def _evaluate(ind):
            layout = self.decoder.decode(ind)
            return (self.fitc.mixed_fitness(layout) + self.pens.penalties(layout),)

        self.toolbox.register("evaluate", _evaluate)

        sel = (selection or "").lower()
        if sel == "tournament":
            self.toolbox.register("select", tools.selTournament, tournsize=3)
        elif sel == "best":
            self.toolbox.register("select", tools.selBest)
        else:
            self.toolbox.register("select", tools.selTournament, tournsize=3)

        cx = (crossover or "PMX").upper()
        if cx == "PMX":
            self.toolbox.register(
                "mate",
                lambda i1, i2: self._index_op_categories(
                    i1, i2, items, tools.cxPartialyMatched
                ),
            )
        elif cx == "OX":
            self.toolbox.register(
                "mate",
                lambda i1, i2: self._index_op_categories(
                    i1, i2, items, tools.cxOrdered
                ),
            )
        else:
            self.toolbox.register(
                "mate",
                lambda i1, i2: self._index_op_categories(i1, i2, items, tools.cxCycle),
            )

        mut = (mutation or "shuffle").lower()
        if mut == "shuffle":
            self.toolbox.register(
                "mutate",
                lambda ind, indpb=0.2: self._index_mut_categories(
                    ind, items, tools.mutShuffleIndexes, indpb=indpb
                ),
                indpb=0.2,
            )
        elif mut == "swap":
            self.toolbox.register(
                "mutate",
                lambda ind, indpb=0.2: self._index_mut_categories(
                    ind, items, tools.mutSwapIndexes, indpb=indpb
                ),
                indpb=0.2,
            )
        else:
            self.toolbox.register(
                "mutate",
                lambda ind: self._index_mut_categories(
                    ind, items, tools.mutInverseIndexes
                ),
            )

    def run(
        self,
        ngen: int = 60,
        pop_size: int = 200,
        seed: Optional[int] = None,
        elite_ratio: float = 0.06,
        adaptive: bool = True,
        return_all: bool = False,
        baseline: Optional[List[str]] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        pop = self.toolbox.population(n=pop_size)

        # FIX: hoàn thiện baseline thành hoán vị đầy đủ
        if baseline:
            baseline_full = _complete_perm_from_seed(self.items, baseline)
            pop[0][:] = baseline_full

        cxpb, mutpb = 0.9, 0.4
        elite_size = max(1, int(round(pop_size * max(0.0, min(0.5, elite_ratio)))))

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(
            lambda ind: ind.fitness.values[0] if ind.fitness.valid else np.nan
        )
        stats.register("avg", np.nanmean)
        stats.register("max", np.nanmax)
        stats.register("min", np.nanmin)
        logbook = []

        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = list(map(self.toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        hof.update(pop)

        for _ in range(ngen):
            if adaptive:
                div = self.diversity(pop)
                if div < 0.4:
                    mutpb = min(1.0, mutpb + 0.1)
                    cxpb = max(0.3, cxpb - 0.1)
                else:
                    mutpb, cxpb = 0.4, 0.9

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(c1, c2)
                    if hasattr(c1.fitness, "values"):
                        del c1.fitness.values
                    if hasattr(c2.fitness, "values"):
                        del c2.fitness.values

            for m in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(m)
                    if hasattr(m.fitness, "values"):
                        del m.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            elites = tools.selBest(pop, elite_size)
            pop[:] = offspring
            pop.sort(key=lambda x: x.fitness.values[0], reverse=True)
            elites.sort(key=lambda x: x.fitness.values[0], reverse=True)
            pop[-elite_size:] = elites

            hof.update(pop)
            record = (
                stats.compile(pop)
                if len(pop)
                else {"avg": np.nan, "max": np.nan, "min": np.nan}
            )
            logbook.append(record)

        best_ind = hof[0]
        best_genome = list(best_ind)
        best_layout = self.decoder.decode(best_genome)
        best_fitness = self.toolbox.evaluate(best_ind)[0]
        if return_all:
            return best_layout, best_fitness, best_genome, logbook, hof, pop
        return best_layout, best_fitness, logbook


# =========================
# Global GA (fallback)
# =========================
class GlobalPermutationGA(_BaseGA):
    """GA trên hoán vị toàn bộ all_items (fallback khi không có tủ mát)."""

    def __init__(
        self,
        items: List[str],
        fitc: FitnessComponents,
        pens: PenaltyRules,
        selection: str = "tournament",
        crossover: str = "PMX",
        mutation: str = "shuffle",
    ):
        _safe_define_deap()
        self.items, self.fitc, self.pens = list(items), fitc, pens
        self.toolbox = base.Toolbox()
        self._register_ops(selection, crossover, mutation)

    def _register_ops(self, selection: str, crossover: str, mutation: str):
        items = self.items

        def _rand_perm():
            g = items[:]
            random.shuffle(g)
            return g

        self.toolbox.register("indices", _rand_perm)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.indices
        )
        # FIX: dùng initRepeat
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        def _evaluate(ind):
            layout = list(ind)
            return (self.fitc.mixed_fitness(layout) + self.pens.penalties(layout),)

        self.toolbox.register("evaluate", _evaluate)

        sel = (selection or "").lower()
        if sel == "tournament":
            self.toolbox.register("select", tools.selTournament, tournsize=3)
        elif sel == "best":
            self.toolbox.register("select", tools.selBest)
        else:
            self.toolbox.register("select", tools.selTournament, tournsize=3)

        cx = (crossover or "PMX").upper()
        if cx == "PMX":
            self.toolbox.register(
                "mate",
                lambda i1, i2: self._index_op_categories(
                    i1, i2, items, tools.cxPartialyMatched
                ),
            )
        elif cx == "OX":
            self.toolbox.register(
                "mate",
                lambda i1, i2: self._index_op_categories(
                    i1, i2, items, tools.cxOrdered
                ),
            )
        else:
            self.toolbox.register(
                "mate",
                lambda i1, i2: self._index_op_categories(i1, i2, items, tools.cxCycle),
            )

        mut = (mutation or "shuffle").lower()
        if mut == "shuffle":
            self.toolbox.register(
                "mutate",
                lambda ind, indpb=0.2: self._index_mut_categories(
                    ind, items, tools.mutShuffleIndexes, indpb=indpb
                ),
                indpb=0.2,
            )
        elif mut == "swap":
            self.toolbox.register(
                "mutate",
                lambda ind, indpb=0.2: self._index_mut_categories(
                    ind, items, tools.mutSwapIndexes, indpb=indpb
                ),
                indpb=0.2,
            )
        else:
            self.toolbox.register(
                "mutate",
                lambda ind: self._index_mut_categories(
                    ind, items, tools.mutInverseIndexes
                ),
            )

    def run(
        self,
        ngen: int = 60,
        pop_size: int = 200,
        seed: Optional[int] = None,
        elite_ratio: float = 0.06,
        adaptive: bool = True,
        return_all: bool = False,
        baseline: Optional[List[str]] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        pop = self.toolbox.population(n=pop_size)

        # FIX: hoàn thiện baseline thành hoán vị đầy đủ
        if baseline:
            baseline_full = _complete_perm_from_seed(self.items, baseline)
            pop[0][:] = baseline_full

        cxpb, mutpb = 0.9, 0.4
        elite_size = max(1, int(round(pop_size * max(0.0, min(0.5, elite_ratio)))))

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(
            lambda ind: ind.fitness.values[0] if ind.fitness.valid else np.nan
        )
        stats.register("avg", np.nanmean)
        stats.register("max", np.nanmax)
        stats.register("min", np.nanmin)
        logbook = []

        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = list(map(self.toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        hof.update(pop)

        for _ in range(ngen):
            if adaptive:
                div = self.diversity(pop)
                if div < 0.4:
                    mutpb = min(1.0, mutpb + 0.1)
                    cxpb = max(0.3, cxpb - 0.1)
                else:
                    mutpb, cxpb = 0.4, 0.9

            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    self.toolbox.mate(c1, c2)
                    if hasattr(c1.fitness, "values"):
                        del c1.fitness.values
                    if hasattr(c2.fitness, "values"):
                        del c2.fitness.values

            for m in offspring:
                if random.random() < mutpb:
                    self.toolbox.mutate(m)
                    if hasattr(m.fitness, "values"):
                        del m.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            elites = tools.selBest(pop, elite_size)
            pop[:] = offspring
            pop.sort(key=lambda x: x.fitness.values[0], reverse=True)
            elites.sort(key=lambda x: x.fitness.values[0], reverse=True)
            pop[-elite_size:] = elites

            hof.update(pop)
            record = (
                stats.compile(pop)
                if len(pop)
                else {"avg": np.nan, "max": np.nan, "min": np.nan}
            )
            logbook.append(record)

        best_ind = hof[0]
        best_layout = list(best_ind)
        best_fitness = self.toolbox.evaluate(best_ind)[0]
        if return_all:
            return best_layout, best_fitness, list(best_ind), logbook, hof, pop
        return best_layout, best_fitness, logbook


# =========================
# Facades
# =========================
class TwoZoneLayoutOptimizer:
    def __init__(
        self,
        all_items: List[str],
        refrig_cats: List[str],
        affinity_matrix: pd.DataFrame,
        forbidden_pairs: Iterable[Tuple[str, str]] = (),
        penalty_forbidden: float = 1000.0,
        hard_rules: Optional[Dict] = None,  # chỉ cần must_together/must_order (nếu có)
        coords: Optional[List[Tuple[float, float]]] = None,
        entr_xy: Optional[Tuple[float, float]] = None,
        cat_support: Optional[Dict[str, float]] = None,
        pairs_list: Optional[List[Tuple[str, str, float]]] = None,
        w_aff: float = 1.0,
        w_pair: float = 0.0,
        w_entr: float = 0.0,
        gamma_support: float = 0.7,
        selection: str = "tournament",
        crossover: str = "PMX",
        mutation: str = "shuffle",
        anchor_start: Optional[int] = None,
    ):
        spec = LayoutSpec(
            all_items=all_items, refrig_cats=refrig_cats, anchor_start=anchor_start
        )
        decoder = DecoderTwoZone(spec)
        fitc = FitnessComponents(
            affinity_matrix=affinity_matrix,
            coords=coords,
            entr_xy=entr_xy,
            cat_support=cat_support,
            pairs_list=pairs_list,
            w_aff=w_aff,
            w_pair=w_pair,
            w_entr=w_entr,
            gamma_support=gamma_support,
        )
        pens = PenaltyRules(
            forbidden_pairs=forbidden_pairs,
            penalty_forbidden=penalty_forbidden,
            hard_rules=hard_rules,
        )
        self.ga = TwoZonePermutationGA(
            spec=spec,
            decoder=decoder,
            fitc=fitc,
            pens=pens,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
        )
        # logbook holders
        self.logbook: Optional[List[Dict]] = None
        self.logbook_df: Optional[pd.DataFrame] = None

    def run(
        self,
        ngen: int = 60,
        pop_size: int = 200,
        seed: Optional[int] = None,
        elite_ratio: float = 0.06,
        adaptive: bool = True,
        return_all: bool = False,
        baseline: Optional[List[str]] = None,
        log_csv_path: Optional[str] = None,  # NEW: lưu CSV nếu muốn
        as_dataframe: bool = True,  # NEW: giữ thêm DataFrame
    ):
        best_layout, best_fitness, logbook = self.ga.run(
            ngen=ngen,
            pop_size=pop_size,
            seed=seed,
            elite_ratio=elite_ratio,
            adaptive=adaptive,
            return_all=False,  # giữ False để trả về bộ 3 (layout, fitness, logbook)
            baseline=baseline,
        )

        # store log
        self.logbook = logbook
        if as_dataframe:
            try:
                self.logbook_df = pd.DataFrame(logbook)
                if log_csv_path:
                    self.logbook_df.to_csv(log_csv_path, index=False)
            except Exception:
                self.logbook_df = None

        return best_layout, best_fitness, logbook


class GlobalLayoutOptimizer:
    def __init__(
        self,
        all_items: List[str],
        affinity_matrix: pd.DataFrame,
        forbidden_pairs: Iterable[Tuple[str, str]] = (),
        penalty_forbidden: float = 1000.0,
        hard_rules: Optional[Dict] = None,
        coords: Optional[List[Tuple[float, float]]] = None,
        entr_xy: Optional[Tuple[float, float]] = None,
        cat_support: Optional[Dict[str, float]] = None,
        pairs_list: Optional[List[Tuple[str, str, float]]] = None,
        w_aff: float = 1.0,
        w_pair: float = 0.0,
        w_entr: float = 0.0,
        gamma_support: float = 0.7,
        selection: str = "tournament",
        crossover: str = "PMX",
        mutation: str = "shuffle",
    ):
        fitc = FitnessComponents(
            affinity_matrix=affinity_matrix,
            coords=coords,
            entr_xy=entr_xy,
            cat_support=cat_support,
            pairs_list=pairs_list,
            w_aff=w_aff,
            w_pair=w_pair,
            w_entr=w_entr,
            gamma_support=gamma_support,
        )
        pens = PenaltyRules(
            forbidden_pairs=forbidden_pairs,
            penalty_forbidden=penalty_forbidden,
            hard_rules=hard_rules,
        )
        self.ga = GlobalPermutationGA(
            items=all_items,
            fitc=fitc,
            pens=pens,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
        )
        # logbook holders
        self.logbook: Optional[List[Dict]] = None
        self.logbook_df: Optional[pd.DataFrame] = None

    def run(
        self,
        ngen: int = 60,
        pop_size: int = 200,
        seed: Optional[int] = None,
        elite_ratio: float = 0.06,
        adaptive: bool = True,
        return_all: bool = False,
        baseline: Optional[List[str]] = None,
        log_csv_path: Optional[str] = None,  # NEW
        as_dataframe: bool = True,  # NEW
    ):
        best_layout, best_fitness, logbook = self.ga.run(
            ngen=ngen,
            pop_size=pop_size,
            seed=seed,
            elite_ratio=elite_ratio,
            adaptive=adaptive,
            return_all=False,
            baseline=baseline,
        )

        # store log
        self.logbook = logbook
        if as_dataframe:
            try:
                self.logbook_df = pd.DataFrame(logbook)
                if log_csv_path:
                    self.logbook_df.to_csv(log_csv_path, index=False)
            except Exception:
                self.logbook_df = None

        return best_layout, best_fitness, logbook
