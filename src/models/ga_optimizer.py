import random

import numpy as np
from deap import algorithms, base, creator, tools


class GeneticLayoutOptimizer:

    def __init__(
        self,
        all_items,
        affinity_matrix,
        forbidden_pairs,
        penalty,
        greedy_ratio=0.2,
        selection="tournament",
        crossover="PMX",
        mutation="shuffle",
        adaptive=True,
        hard_rules=None,
    ):
        self.all_items = all_items
        self.affinity_matrix = affinity_matrix
        self.forbidden_pairs = forbidden_pairs
        self.penalty = penalty
        self.greedy_ratio = greedy_ratio
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.adaptive = adaptive
        self.hard_rules = hard_rules or {}

    @staticmethod
    def create_valid_individual(all_items, hard_rules):
        """Sinh ra một layout hợp lệ theo hard_rules đã cho."""
        items = list(all_items)
        for cat in hard_rules.get("must_at_entry", []):
            if cat in items:
                items.remove(cat)
        for cat in hard_rules.get("must_at_exit", []):
            if cat in items:
                items.remove(cat)
        random.shuffle(items)
        layout = []
        layout.extend(hard_rules.get("must_at_entry", []))
        layout.extend(items)
        layout.extend(hard_rules.get("must_at_exit", []))
        # must_together: đảm bảo cặp luôn liền kề (sau a là b)
        for a, b in hard_rules.get("must_together", []):
            if a in layout and b in layout:
                idx_a = layout.index(a)
                idx_b = layout.index(b)
                if abs(idx_a - idx_b) != 1:
                    layout.pop(idx_b)
                    idx_a = layout.index(a)
                    layout.insert(idx_a + 1, b)
        # must_order: đảm bảo a trước b
        for a, b in hard_rules.get("must_order", []):
            if a in layout and b in layout:
                idx_a = layout.index(a)
                idx_b = layout.index(b)
                if idx_a > idx_b:
                    layout.pop(idx_a)
                    idx_b = layout.index(b)
                    layout.insert(idx_b, a)
        return layout

    def layout_fitness(self, layout):
        return sum(
            self.affinity_matrix.loc[layout[i], layout[i + 1]]
            for i in range(len(layout) - 1)
        )

    def eval_layout(self, individual):
        layout = [self.all_items[i] for i in individual]
        penalty_score = 0
        # Forbidden pairs
        for i in range(len(layout) - 1):
            a, b = layout[i], layout[i + 1]
            if (a, b) in self.forbidden_pairs:
                penalty_score -= self.penalty
        # Hard rules
        rules = self.hard_rules
        for item in rules.get("must_at_entry", []):
            if layout[0] != item:
                penalty_score -= 9999
        for item in rules.get("must_at_exit", []):
            if layout[-1] != item:
                penalty_score -= 9999
        for a, b in rules.get("must_together", []):
            if abs(layout.index(a) - layout.index(b)) != 1:
                penalty_score -= 9999
        for a, b in rules.get("must_order", []):
            if layout.index(a) > layout.index(b):
                penalty_score -= 9999
        return (self.layout_fitness(layout) + penalty_score,)

    def compute_diversity(self, population):
        as_set = {tuple(ind) for ind in population}
        return len(as_set) / len(population)

    def run(
        self,
        ngen=50,
        pop_size=200,
        greedy_layout=None,
        seed=None,
        record_logbook=True,
        return_all=False,
        init_population_extra=None,  # Thêm tham số seed population
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        indices_list = list(range(len(self.all_items)))
        for cname in ["FitnessMax", "Individual"]:
            if hasattr(creator, cname):
                delattr(creator, cname)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, indices_list, len(self.all_items))
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.indices
        )
        toolbox.register("population", list, toolbox.individual)
        toolbox.register("evaluate", self.eval_layout)

        # Selection
        if self.selection == "tournament":
            toolbox.register("select", tools.selTournament, tournsize=3)
        elif self.selection == "roulette":
            toolbox.register("select", tools.selRoulette)
        else:
            toolbox.register("select", tools.selTournament, tournsize=3)

        # Crossover
        if self.crossover == "PMX":
            toolbox.register("mate", tools.cxPartialyMatched)
        elif self.crossover == "OX":
            toolbox.register("mate", tools.cxOrdered)
        elif self.crossover == "CX":
            toolbox.register("mate", tools.cxCycle)
        else:
            toolbox.register("mate", tools.cxPartialyMatched)

        # Mutation
        if self.mutation == "shuffle":
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        elif self.mutation == "swap":
            toolbox.register("mutate", tools.mutSwapIndexes, indpb=0.2)
        elif self.mutation == "inversion":
            toolbox.register("mutate", tools.mutInverseIndexes)
        else:
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

        n_greedy = int(pop_size * self.greedy_ratio)
        pop = []

        # ==== Khởi tạo population hợp lệ ====
        if self.hard_rules and n_greedy > 0:
            for _ in range(n_greedy):
                layout = self.create_valid_individual(self.all_items, self.hard_rules)
                indices = [self.all_items.index(cat) for cat in layout]
                pop.append(creator.Individual(indices))
        elif greedy_layout is not None and n_greedy > 0:
            greedy_indices = [self.all_items.index(cat) for cat in greedy_layout]
            pop.append(creator.Individual(greedy_indices.copy()))
            for _ in range(n_greedy - 1):
                ind = greedy_indices.copy()
                random.shuffle(ind)
                pop.append(creator.Individual(ind))

        # --------- Thêm cá thể khởi tạo từ ngoài (ví dụ: layout thực tế) ---------
        if init_population_extra:
            for layout in init_population_extra:
                indices = [self.all_items.index(cat) for cat in layout]
                pop.append(creator.Individual(indices))
        # -------------------------------------------------------------------------

        # Tiếp tục random cho đến khi đủ pop_size
        pop += [toolbox.individual() for _ in range(pop_size - len(pop))]

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(
            lambda ind: ind.fitness.values[0] if len(ind.fitness.values) > 0 else np.nan
        )
        stats.register("avg", np.nanmean)
        stats.register("max", np.nanmax)
        stats.register("min", np.nanmin)
        diversity_log = []

        cxpb, mutpb = 0.9, 0.4  # base rate

        logbook = []
        for gen in range(ngen):
            # Đánh giá cá thể
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            diversity = self.compute_diversity(pop)
            diversity_log.append(diversity)

            # Adaptive mutation/crossover
            if self.adaptive:
                if diversity < 0.4:
                    mutpb = min(1.0, mutpb + 0.1)
                    cxpb = max(0.3, cxpb - 0.1)
                else:
                    mutpb = 0.4
                    cxpb = 0.9

            # Chọn, lai ghép, đột biến
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            pop[:] = offspring
            hof.update(pop)

            # Lọc valid cá thể trước khi compile statistics
            valid_pop = [ind for ind in pop if len(ind.fitness.values) > 0]
            if len(valid_pop) > 0:
                record = stats.compile(valid_pop)
            else:
                record = {"avg": np.nan, "max": np.nan, "min": np.nan}
            record["diversity"] = diversity
            logbook.append(record)

        best_indices = hof[0]
        best_layout = [self.all_items[i] for i in best_indices]
        best_fitness = self.layout_fitness(best_layout)
        if return_all:
            return best_layout, best_fitness, logbook, diversity_log, hof, pop
        if record_logbook:
            return best_layout, best_fitness, logbook
        return best_layout, best_fitness

    def run_ensemble(
        self, ngen=50, pop_size=200, greedy_layout=None, n_runs=5, **kwargs
    ):
        results = []
        for run in range(n_runs):
            seed = random.randint(0, 100000)
            best_layout, best_fitness, logbook = self.run(
                ngen=ngen,
                pop_size=pop_size,
                greedy_layout=greedy_layout,
                seed=seed,
                record_logbook=True,
                **kwargs  # <--- truyền init_population_extra qua đây luôn
            )
            results.append(
                {
                    "seed": seed,
                    "best_layout": best_layout,
                    "best_fitness": best_fitness,
                    "logbook": logbook,
                }
            )
        best_run = max(results, key=lambda x: x["best_fitness"])
        return best_run, results
