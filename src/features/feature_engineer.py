from collections import Counter

import pandas as pd

from src.utils.general_utils import safe_to_list


class FeatureEngineer:
    def __init__(self, assoc_rules, freq_itemsets):
        self.ante_count = Counter(
            sum(assoc_rules["antecedent"].apply(safe_to_list), [])
        )
        self.cons_count = Counter(
            sum(assoc_rules["consequent"].apply(safe_to_list), [])
        )
        self.all_categories = sorted(
            set(self.ante_count.keys()) | set(self.cons_count.keys())
        )

    def engineer(self, price_df: pd.DataFrame) -> pd.DataFrame:
        df = price_df.copy()
        df = df[df["item_id"].isin(self.all_categories)]
        df["Date"] = pd.to_datetime(df["Date"])
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["combo_role"] = df["item_id"].apply(self.infer_combo_role)
        df = pd.get_dummies(df, columns=["combo_role"], drop_first=True, dtype=int)
        return df

    def infer_combo_role(self, item):
        if self.ante_count.get(item, 0) > self.cons_count.get(item, 0):
            return "leader"
        elif self.cons_count.get(item, 0) > self.ante_count.get(item, 0):
            return "cross_sell"
        return "other"
