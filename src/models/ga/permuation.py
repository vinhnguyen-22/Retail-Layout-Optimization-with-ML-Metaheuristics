import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from deap import base, creator, tools

from src.models.ga.fitness import FitnessComponents
from src.models.ga.helper import BaseGA, complete_perm_from_seed, safe_define_deap
from src.models.ga.penalties import PenaltyRules


class PermutationGA(BaseGA):
    """GA trên hoán vị; hành vi decode phụ thuộc decoder (TwoZone/Identity)."""

    def __init__(
        self,
        items: List[str],
        decoder,  # DecoderTwoZone() hoặc IdentityDecoder()
        fitc: FitnessComponents,
        pens: PenaltyRules,
        selection: str = "tournament",
        crossover: str = "PMX",
        mutation: str = "shuffle",
    ):
        safe_define_deap()
        self.items, self.decoder, self.fitc, self.pens = (
            list(items),
            decoder,
            fitc,
            pens,
        )
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

        if baseline:
            baseline_full = complete_perm_from_seed(self.items, baseline)
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
        best_layout = self.decoder.decode(list(best_ind))
        best_fitness = self.toolbox.evaluate(best_ind)[0]
        if return_all:
            return best_layout, best_fitness, list(best_ind), logbook, hof, pop
        return best_layout, best_fitness, logbook
