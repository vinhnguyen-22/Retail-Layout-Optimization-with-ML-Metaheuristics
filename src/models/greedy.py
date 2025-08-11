import random


class GreedyLayout:
    def __init__(self, all_items):
        self.all_items = all_items

    def init_layout(self, affinity):
        categories_left = set(self.all_items)
        start = max(categories_left, key=lambda cat: affinity.loc[cat, :].sum())
        layout = [start]
        categories_left.remove(start)
        while categories_left:
            last = layout[-1]
            # Nếu tất cả affinity = 0, random chọn
            if all(affinity.loc[last, cat] == 0 for cat in categories_left):
                next_cat = random.choice(list(categories_left))
            else:
                next_cat = max(categories_left, key=lambda cat: affinity.loc[last, cat])
            layout.append(next_cat)
            categories_left.remove(next_cat)
        return layout

    def local_search(self, layout, affinity, max_iter=20):
        layout = layout.copy()
        best_score = self.layout_fitness(layout, affinity)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            for i in range(1, len(layout) - 2):
                for j in range(i + 1, len(layout) - 1):
                    new_layout = layout.copy()
                    new_layout[i], new_layout[j] = new_layout[j], new_layout[i]
                    new_score = self.layout_fitness(new_layout, affinity)
                    if new_score > best_score:
                        layout = new_layout
                        best_score = new_score
                        improved = True
            it += 1
        return layout

    def layout_fitness(self, layout, affinity):
        return sum(
            affinity.loc[layout[i], layout[i + 1]] for i in range(len(layout) - 1)
        )
