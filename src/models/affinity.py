import numpy as np
import pandas as pd


class AffinityBuilder:
    def __init__(self, assoc_rules, freq_itemsets, all_items, margin_matrix=None):
        self.assoc_rules = assoc_rules
        self.freq_itemsets = freq_itemsets
        self.all_items = all_items
        self.margin_matrix = margin_matrix

    def build_affinity(self, lift_threshold, w_lift=0.5, w_conf=0.5, w_margin=0.0):
        affinity_lift = pd.DataFrame(0.0, index=self.all_items, columns=self.all_items)
        affinity_conf = pd.DataFrame(0.0, index=self.all_items, columns=self.all_items)
        for _, row in self.assoc_rules.iterrows():
            ants = eval(row["antecedent"])
            cons = eval(row["consequent"])
            lift = row["lift"]
            conf = row["confidence"]
            if lift >= lift_threshold:
                for a in ants:
                    for c in cons:
                        affinity_lift.loc[a, c] = max(affinity_lift.loc[a, c], lift)
                        affinity_lift.loc[c, a] = max(affinity_lift.loc[c, a], lift)
                        affinity_conf.loc[a, c] = max(affinity_conf.loc[a, c], conf)
                        affinity_conf.loc[c, a] = max(affinity_conf.loc[c, a], conf)
        affinity_lift = self.normalize(affinity_lift, method="minmax", clip_val=5.0)
        affinity_conf = self.normalize(affinity_conf, method="minmax")
        if w_margin > 0 and self.margin_matrix is not None:
            affinity_margin = self.normalize(self.margin_matrix, method="minmax")
        else:
            affinity_margin = pd.DataFrame(
                0, index=self.all_items, columns=self.all_items
            )
        affinity = (
            w_lift * affinity_lift + w_conf * affinity_conf + w_margin * affinity_margin
        )
        if w_lift + w_conf + w_margin > 0:
            affinity = affinity / (w_lift + w_conf + w_margin)
        return affinity

    @staticmethod
    def normalize(df, method="minmax", clip_val=None):
        df = df.copy()
        if clip_val is not None:
            df = df.clip(upper=clip_val)
        if method == "minmax":
            minv, maxv = df.values.min(), df.values.max()
            if maxv > minv:
                df = (df - minv) / (maxv - minv)
        elif method == "zscore":
            mean, std = df.values.mean(), df.values.std()
            if std > 0:
                df = (df - mean) / std
        return df

    @staticmethod
    def kernelize(df, gamma=1.0):
        return np.exp(-gamma * (1 - df))
