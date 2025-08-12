from typing import List

from deap import base, creator


def complete_perm_from_seed(all_items: List[str], seed_order: List[str]) -> List[str]:
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


def safe_define_deap():
    for cname in ["FitnessMax", "Individual"]:
        if hasattr(creator, cname):
            delattr(creator, cname)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    creator.create("Individual", list, fitness=creator.FitnessMax)


class BaseGA:
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
