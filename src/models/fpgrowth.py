import numpy as np
import pandas as pd
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import array_distinct, collect_list

from src.utils.general_utils import array_to_list
from src.utils.rule_metrics import RuleMetricsCalculator


class FPGrowthRunner:
    def __init__(
        self,
        transactions,
        min_support=0.001,
        min_confidence=0.5,
        min_lift=1.0,
        max_rules=10000,
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_rules = max_rules
        self.transactions = transactions

    def prepare_data(self):
        transactions = self.transactions.groupBy("MergedId").agg(
            array_distinct(collect_list("Item_id")).alias("items")
        )
        return transactions

    def run(self):
        transactions = self.prepare_data()
        fp_growth = FPGrowth(
            itemsCol="items",
            minSupport=self.min_support,
            minConfidence=self.min_confidence,
        )
        model = fp_growth.fit(transactions)
        frequent_itemsets = model.freqItemsets
        rules = model.associationRules
        return frequent_itemsets, rules

    def process(
        self,
        frequent_itemsets,
        rules,
        save_dir,
    ):
        # ---- Tính toán support cho itemsets ----
        total_transactions = self.transactions.count()
        frequent_items_df = frequent_itemsets.toPandas()
        frequent_items_df["support"] = frequent_items_df["freq"] / total_transactions
        frequent_items_df["items"] = frequent_items_df["items"].apply(
            lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [str(x)]
        )

        # ---- Lọc rule và export ra parquet ----
        rules_filtered = rules.filter(
            f"confidence >= {self.min_confidence} AND lift >= {self.min_lift}"
        ).limit(self.max_rules)
        parquet_path = save_dir / "association_rules_filtered.parquet"
        rules_filtered.write.mode("overwrite").parquet(str(parquet_path))
        rules_df = pd.read_parquet(parquet_path)

        # ---- Chuyển array -> list cho rule ----
        rules_df["antecedent"] = rules_df["antecedent"].apply(array_to_list)
        rules_df["consequent"] = rules_df["consequent"].apply(array_to_list)

        # ---- Support map để tính metrics ----
        support_map = {
            tuple(sorted(row["items"])): row["support"]
            for _, row in frequent_items_df.iterrows()
        }

        def get_support(item_list):
            key = tuple(sorted(item_list))
            return support_map.get(key, np.nan)

        rules_df["support_antecedent"] = rules_df["antecedent"].apply(get_support)
        rules_df["support_consequent"] = rules_df["consequent"].apply(get_support)

        # ---- Tính metrics nâng cao ----
        rule = RuleMetricsCalculator()
        rules_df = rule.calc_metric(rules_df)
        rules_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        frequent_items_df_export = frequent_items_df[
            frequent_items_df["items"].apply(len) >= 2
        ].copy()
        frequent_items_df_export["items"] = frequent_items_df_export["items"].apply(str)
        frequent_items_df_export.to_csv(
            str(save_dir / "frequent_itemsets.csv"), index=False
        )
        rules_df["antecedent"] = rules_df["antecedent"].apply(lambda l: str(l))
        rules_df["consequent"] = rules_df["consequent"].apply(lambda l: str(l))
        rules_df.to_csv(str(save_dir / "association_rules.csv"), index=False)
        return frequent_items_df_export, rules_df
