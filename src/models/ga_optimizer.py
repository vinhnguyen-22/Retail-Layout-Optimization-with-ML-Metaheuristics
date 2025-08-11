import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from deap import base, creator, tools


def euclid(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


class GeneticLayoutOptimizer:
    """
    GA cho tối ưu hoán vị layout (permutation GA) với hard-rules & forbidden pairs.

    - Biểu diễn cá thể: danh sách chỉ số (int) tương ứng vị trí trong self.all_items.
    - Fitness: maximize (tổng affinity liền kề) + penalty (âm) cho vi phạm rule.
    - Bảo toàn hoán vị: dùng PMX/OX/CX; mutation: shuffle/swap/inversion.
    - Elitism: giữ lại top-k mỗi thế hệ.
    - Adaptive rates: điều chỉnh cx/mut theo đa dạng quần thể.
    - Repair nhẹ sau lai/đột biến để giảm vi phạm hard-rules.

    Tham số:
        all_items: List[str]                 # danh sách Category theo chỉ số cố định
        affinity_matrix: pd.DataFrame        # ma trận affinity (index/columns là Category)
        forbidden_pairs: set[(catA, catB)]   # cặp bị cấm liền kề (không định hướng)
        penalty: float                       # điểm phạt cho mỗi vi phạm cặp cấm
        greedy_ratio: float                  # tỉ lệ seed cá thể "có định hướng"
        selection: {"tournament","roulette","best"} (mặc định tournament)
        crossover: {"PMX","OX","CX"} (mặc định PMX)
        mutation: {"shuffle","swap","inversion"} (mặc định shuffle)
        adaptive: bool                       # bật/tắt adaptive rates
        hard_rules: dict                     # {"must_together":[(a,b),...],
                                             #  "must_order":[(a,b),...],
                                             #  "must_group_refrigerated":[cat,...]}
    """

    def __init__(
        self,
        all_items: List[str],
        affinity_matrix,  # pd.DataFrame [cat x cat]
        forbidden_pairs: Iterable[Tuple[str, str]],
        penalty: float,
        greedy_ratio: float = 0.2,
        selection: str = "tournament",
        crossover: str = "PMX",
        mutation: str = "shuffle",
        adaptive: bool = True,
        hard_rules: Optional[Dict] = None,
        # ----- NEW: mixed-fitness params -----
        coords: Optional[
            List[Tuple[float, float]]
        ] = None,  # slot coords theo thứ tự (y,x)
        entr_xy: Optional[Tuple[float, float]] = None,
        cat_support: Optional[Dict[str, float]] = None,  # support của từng category
        pairs_list: Optional[List[Tuple[str, str, float]]] = None,  # (a,b,w)
        w_aff: float = 1.0,
        w_pair: float = 0.0,
        w_entr: float = 0.0,
        gamma_support: float = 0.7,
    ):
        self.all_items = list(all_items)
        self.affinity_matrix = affinity_matrix
        self.forbidden_pairs = set(forbidden_pairs)
        self.penalty = float(penalty)
        self.greedy_ratio = float(greedy_ratio)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.adaptive = adaptive
        self.hard_rules = hard_rules or {}

        # mixed
        self.coords = list(coords) if coords is not None else None
        self.entr_xy = entr_xy
        self.cat_support = cat_support or {}
        self.pairs_list = pairs_list or []
        self.w_aff = float(w_aff)
        self.w_pair = float(w_pair)
        self.w_entr = float(w_entr)
        self.gamma_support = float(gamma_support)

        self._cat2idx = {c: i for i, c in enumerate(self.all_items)}
        self._sanitize_hard_rules()

        # mean distance (chuẩn hóa)
        if self.coords is not None and len(self.coords) >= 2:
            m = min(1000, len(self.coords) * 2)
            idx = np.random.randint(0, len(self.coords), size=(m, 2))
            d = [euclid(self.coords[i], self.coords[j]) for i, j in idx if i != j]
            self.mean_dist = float(np.mean(d)) if d else 1.0
        else:
            self.mean_dist = 1.0

    # ---------------- helpers ----------------
    def _sanitize_hard_rules(self):
        known = set(self.all_items)
        out = {}
        for k, v in (self.hard_rules or {}).items():
            if k in ("must_together", "must_order"):
                out[k] = [(a, b) for (a, b) in v if a in known and b in known]
            elif k == "must_group_refrigerated":
                out[k] = [c for c in v if c in known]
            else:
                out[k] = v
        self.hard_rules = out

    def _layout_from_indices(self, indices: List[int]) -> List[str]:
        return [self.all_items[i] for i in indices]

    def _is_valid_perm(self, ind: List[int]) -> bool:
        return len(ind) == len(self.all_items) and sorted(ind) == list(
            range(len(self.all_items))
        )

    # ---------------- fitness parts ----------------
    def _score_affinity_adjacent(self, layout_cats: List[str]) -> float:
        # Nếu có coords, chỉ tính adjacency trên phần layout gắn vào slot thật
        if self.coords is not None:
            L = min(len(layout_cats), len(self.coords))
        else:
            L = len(layout_cats)
        s = 0.0
        for i in range(max(0, L - 1)):
            a, b = layout_cats[i], layout_cats[i + 1]
            s += float(self.affinity_matrix.loc[a, b])
        # Nếu không có coords (pure association), L == len(layout_cats) như cũ
        return s

    def _cost_pairs_distance(self, layout_cats: List[str]) -> float:
        """∑ w_ab * dist(slot(a), slot(b)) / mean_dist"""
        if not self.coords or not self.pairs_list:
            return 0.0
        N = min(len(layout_cats), len(self.coords))
        if N <= 1:
            return 0.0
        # chỉ map các cat thuộc N slot đầu
        pos = {layout_cats[i]: i for i in range(N)}
        tot = 0.0
        for a, b, w in self.pairs_list:
            ia = pos.get(a)
            ib = pos.get(b)
            if ia is None or ib is None:
                continue
            tot += float(w) * euclid(self.coords[ia], self.coords[ib])
        return tot / max(1e-9, self.mean_dist)

    def _cost_entrance_distance(self, layout_cats: List[str]) -> float:
        """∑ support(c)^gamma * dist(entr, slot(c)) / (mean_dist*N)"""
        if not self.coords or self.entr_xy is None:
            return 0.0
        N = min(len(layout_cats), len(self.coords))
        if N == 0:
            return 0.0
        tot = 0.0
        for i in range(N):
            c = layout_cats[i]
            s = float(self.cat_support.get(c, 0.0)) ** self.gamma_support
            tot += s * euclid(self.entr_xy, self.coords[i])
        return tot / max(1e-9, self.mean_dist * N)

    # ---------------- penalties ----------------
    def _penalties(self, layout_cats: List[str]) -> float:
        pen = 0.0
        # forbidden kề nhau (2 chiều)
        for i in range(len(layout_cats) - 1):
            a, b = layout_cats[i], layout_cats[i + 1]
            if (a, b) in self.forbidden_pairs or (b, a) in self.forbidden_pairs:
                pen -= self.penalty
        # hard rules mạnh
        for a, b in self.hard_rules.get("must_together", []):
            if a in layout_cats and b in layout_cats:
                if abs(layout_cats.index(a) - layout_cats.index(b)) != 1:
                    pen -= 9_999
        for a, b in self.hard_rules.get("must_order", []):
            if a in layout_cats and b in layout_cats:
                if layout_cats.index(a) > layout_cats.index(b):
                    pen -= 9_999
        refrig = self.hard_rules.get("must_group_refrigerated", [])
        if refrig:
            idxs = [layout_cats.index(c) for c in refrig if c in layout_cats]
            if idxs and (max(idxs) - min(idxs) + 1 != len(idxs)):
                pen -= 9_999
        return pen

    def _full_perm_from_seed(self, seed_layout: List[str]) -> List[int]:
        """
        Từ seed (danh sách Category, có thể thiếu hoặc lẫn mục lạ),
        tạo hoán vị đủ độ dài theo self.all_items:
        - giữ nguyên thứ tự các phần tử seed hợp lệ (có trong all_items)
        - phần còn lại bổ sung ngẫu nhiên phía sau
        """
        # map cat -> index đã có: self._cat2idx
        seen = set()
        seed_indices = []
        for c in seed_layout or []:
            idx = self._cat2idx.get(c)
            if idx is not None and idx not in seen:
                seed_indices.append(idx)
                seen.add(idx)

        rest = [i for i in range(len(self.all_items)) if i not in seen]
        random.shuffle(rest)
        return seed_indices + rest

    # ===== Repairs (làm mềm hard-rules) =====

    def _repair_must_together(self, ind: list):
        """Đảm bảo (a,b) liền kề: đưa b ngay sau a, giữ relative order tối thiểu."""
        rules = self.hard_rules.get("must_together", [])
        if not rules:
            return
        for a, b in rules:
            ia = self._cat2idx.get(a)
            ib = self._cat2idx.get(b)
            if ia is None or ib is None:
                continue
            try:
                pa = ind.index(ia)
                pb = ind.index(ib)
            except ValueError:
                continue
            if abs(pa - pb) != 1:
                # đưa b về ngay sau a
                ind.pop(pb)
                pa = ind.index(ia)  # cập nhật vì list đã thay đổi
                ind.insert(pa + 1, ib)

    def _repair_must_order(self, ind: list):
        """Đảm bảo a đứng trước b: nếu sai thứ tự, kéo a lên trước b."""
        rules = self.hard_rules.get("must_order", [])
        if not rules:
            return
        for a, b in rules:
            ia = self._cat2idx.get(a)
            ib = self._cat2idx.get(b)
            if ia is None or ib is None:
                continue
            try:
                pa = ind.index(ia)
                pb = ind.index(ib)
            except ValueError:
                continue
            if pa > pb:
                # kéo a lên ngay trước b
                ind.pop(pa)
                pb = ind.index(ib)  # vị trí b có thể đã thay đổi
                ind.insert(pb, ia)

    def _repair_group_refrigerated(self, ind: list):
        """Nén các category tủ mát thành một block liên tiếp, giữ thứ tự hiện tại."""
        refrig = self.hard_rules.get("must_group_refrigerated", [])
        if not refrig:
            return
        cold_vals = [self._cat2idx[c] for c in refrig if c in self._cat2idx]
        if not cold_vals:
            return
        # vị trí hiện tại của các item lạnh
        try:
            positions = [ind.index(v) for v in cold_vals]
        except ValueError:
            # có item không nằm trong hoán vị (không xảy ra với permutation hợp lệ), bỏ qua
            return
        if len(positions) <= 1:
            return
        if max(positions) - min(positions) + 1 == len(positions):
            return  # đã liên tiếp rồi

        insert_at = min(positions)
        ordered_block = [ind[p] for p in sorted(positions)]  # giữ thứ tự hiện có
        # xoá block lạnh khỏi ind
        cold_set = set(ordered_block)
        ind[:] = [x for x in ind if x not in cold_set]
        # chèn lại thành block liên tiếp
        for off, val in enumerate(ordered_block):
            ind.insert(insert_at + off, val)

    def _repair_all(self, ind: list):
        """Áp dụng lần lượt các sửa chữa nhẹ để giảm vi phạm hard-rules."""
        if not self.hard_rules:
            return
        self._repair_must_order(ind)
        self._repair_must_together(ind)
        self._repair_group_refrigerated(ind)

    def _repair_all(self, ind: list):
        """Áp dụng lần lượt các sửa chữa nhẹ để giảm vi phạm hard-rules."""
        if not self.hard_rules:
            return
        self._repair_must_order(ind)
        self._repair_must_together(ind)
        self._repair_group_refrigerated(ind)

    @staticmethod
    def compute_diversity(population):
        """
        Đo đa dạng quần thể rất gọn: tỉ lệ cá thể duy nhất / tổng số cá thể.
        population là list các cá thể (list[int]).
        """
        if not population:
            return 0.0
        # dùng tuple để hash
        uniq = {tuple(ind) for ind in population}
        return len(uniq) / len(population)

    # ---------------- DEAP hooks ----------------
    def eval_layout(self, individual: List[int]):
        layout_cats = self._layout_from_indices(individual)
        score_aff = self._score_affinity_adjacent(layout_cats)
        cost_pair = self._cost_pairs_distance(layout_cats)
        cost_ent = self._cost_entrance_distance(layout_cats)
        pen = self._penalties(layout_cats)
        # maximize:
        mixed = (
            self.w_aff * score_aff
            - self.w_pair * cost_pair
            - self.w_entr * cost_ent
            + pen
        )
        return (mixed,)

    # =======================
    # GA main
    # =======================
    def run(
        self,
        ngen: int = 50,
        pop_size: int = 200,
        greedy_layout: Optional[List[str]] = None,
        seed: Optional[int] = None,
        record_logbook: bool = True,
        return_all: bool = False,
        init_population_extra: Optional[List[List[str]]] = None,
        elite_ratio: float = 0.05,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n_items = len(self.all_items)
        indices_all = list(range(n_items))

        # Re-create DEAP classes safely (tránh xung đột khi import nhiều lần)
        for cname in ["FitnessMax", "Individual"]:
            if hasattr(creator, cname):
                delattr(creator, cname)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        # Khởi tạo cá thể hợp lệ ngẫu nhiên
        toolbox.register("indices", random.sample, indices_all, n_items)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.indices
        )
        toolbox.register("population", list, toolbox.individual)
        toolbox.register("evaluate", self.eval_layout)

        # Selection
        sel = (self.selection or "").lower()
        if sel == "tournament":
            toolbox.register("select", tools.selTournament, tournsize=3)
        elif sel == "best":
            toolbox.register("select", tools.selBest)
        else:
            # roulette dễ lỗi với fitness âm -> fallback tournament
            toolbox.register("select", tools.selTournament, tournsize=3)

        # Crossover
        cx = (self.crossover or "PMX").upper()
        if cx == "PMX":
            toolbox.register("mate", tools.cxPartialyMatched)
        elif cx == "OX":
            toolbox.register("mate", tools.cxOrdered)
        elif cx == "CX":
            toolbox.register("mate", tools.cxCycle)
        else:
            toolbox.register("mate", tools.cxPartialyMatched)

        # Mutation
        mut = (self.mutation or "shuffle").lower()
        if mut == "shuffle":
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        elif mut == "swap":
            toolbox.register("mutate", tools.mutSwapIndexes, indpb=0.2)
        elif mut == "inversion":
            toolbox.register("mutate", tools.mutInverseIndexes)
        else:
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

        # ======= Population init =======
        pop: List[creator.Individual] = []

        # seed theo hard_rules (optional): tạo layout hợp lệ rồi index hoá
        n_greedy = int(pop_size * max(0.0, min(1.0, self.greedy_ratio)))
        if n_greedy > 0 and greedy_layout is not None:
            # 1 cá thể seed từ greedy_layout + biến thể
            base_seed = self._full_perm_from_seed(greedy_layout)
            pop.append(creator.Individual(base_seed))
            for _ in range(n_greedy - 1):
                tmp = base_seed.copy()
                random.shuffle(tmp)
                pop.append(creator.Individual(tmp))
        elif n_greedy > 0 and self.hard_rules:
            # nếu không có greedy_layout, nhưng có hard_rules: sinh cá thể random rồi repair
            for _ in range(n_greedy):
                ind = creator.Individual(random.sample(indices_all, n_items))
                self._repair_all(ind)
                pop.append(ind)

        # thêm cá thể seed từ ngoài (layout category)
        if init_population_extra:
            seen = set()
            for lay in init_population_extra:
                idxs = self._full_perm_from_seed(lay)
                t = tuple(idxs)
                if t in seen:
                    continue
                seen.add(t)
                pop.append(creator.Individual(idxs))

        # bổ sung random cho đủ pop_size
        while len(pop) < pop_size:
            pop.append(toolbox.individual())

        # đảm bảo hợp lệ (DEAP operators sẽ giữ hoán vị hợp lệ)
        for i, ind in enumerate(pop):
            if not self._is_valid_perm(ind):
                pop[i] = creator.Individual(random.sample(indices_all, n_items))

        # ======= GA settings =======
        cxpb, mutpb = 0.9, 0.4
        elite_size = max(1, int(round(pop_size * max(0.0, min(0.5, elite_ratio)))))

        # Stats & HOF
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(
            lambda ind: ind.fitness.values[0] if ind.fitness.valid else np.nan
        )
        stats.register("avg", np.nanmean)
        stats.register("max", np.nanmax)
        stats.register("min", np.nanmin)

        logbook = []
        diversity_log = []

        # ======= Evolution loop =======
        # Đánh giá quần thể ban đầu
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        hof.update(pop)

        for gen in range(ngen):
            # Diversity & adaptive
            diversity = self.compute_diversity(pop)
            diversity_log.append(diversity)
            if self.adaptive:
                if diversity < 0.4:
                    mutpb = min(1.0, mutpb + 0.1)
                    cxpb = max(0.3, cxpb - 0.1)
                else:
                    mutpb, cxpb = 0.4, 0.9

            # Chọn & nhân bản
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Lai ghép
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(c1, c2)
                    if hasattr(c1.fitness, "values"):
                        del c1.fitness.values
                    if hasattr(c2.fitness, "values"):
                        del c2.fitness.values

            # Đột biến
            for mut_ind in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mut_ind)
                    if hasattr(mut_ind.fitness, "values"):
                        del mut_ind.fitness.values

            # Repair nhẹ để giảm vi phạm hard-rules
            for mut_ind in offspring:
                self._repair_all(mut_ind)

            # Đánh giá offspring trước khi elitism/HOF
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            # Elitism
            elites = tools.selBest(pop, elite_size)
            pop[:] = offspring
            # chèn elite ghi đè lên các cá thể tệ nhất
            pop.sort(key=lambda x: x.fitness.values[0], reverse=True)
            elites.sort(key=lambda x: x.fitness.values[0], reverse=True)
            pop[-elite_size:] = elites

            hof.update(pop)

            # Ghi thống kê
            record = (
                stats.compile(pop)
                if len(pop)
                else {"avg": np.nan, "max": np.nan, "min": np.nan}
            )
            record["diversity"] = diversity
            logbook.append(record)

        best_ind = hof[0]
        best_layout = self._layout_from_indices(best_ind)
        # dùng eval để bảo đảm tính lại (an toàn)
        best_fitness = self.eval_layout(best_ind)[0]

        if return_all:
            return best_layout, best_fitness, logbook, diversity_log, hof, pop
        if record_logbook:
            return best_layout, best_fitness, logbook
        return best_layout, best_fitness

    def run_ensemble(
        self,
        ngen: int = 50,
        pop_size: int = 200,
        greedy_layout: Optional[List[str]] = None,
        n_runs: int = 5,
        seed: Optional[int] = None,
        **kwargs,
    ):
        results = []
        base_seed = seed if seed is not None else random.randint(0, 10**6)
        for r in range(n_runs):
            run_seed = base_seed + r
            best_layout, best_fitness, logbook = self.run(
                ngen=ngen,
                pop_size=pop_size,
                greedy_layout=greedy_layout,
                seed=run_seed,
                record_logbook=True,
                **kwargs,
            )
            results.append(
                {
                    "seed": run_seed,
                    "best_layout": best_layout,
                    "best_fitness": best_fitness,
                    "logbook": logbook,
                }
            )
        best_run = max(results, key=lambda x: x["best_fitness"])
        return best_run, results
