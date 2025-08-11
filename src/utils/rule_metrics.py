import re

import numpy as np


class RuleMetricsCalculator:
    @staticmethod
    def leverage(support, support_a, support_c):
        return support - (support_a * support_c)

    @staticmethod
    def conviction(support_c, confidence):
        if confidence >= 1.0:
            return np.inf
        return (1 - support_c) / (1 - confidence)

    @staticmethod
    def all_confidence(support, support_a, support_c):
        return (
            support / max(support_a, support_c) if max(support_a, support_c) > 0 else 0
        )

    @staticmethod
    def interest(support, support_a, support_c):
        return support / (support_a * support_c) if (support_a * support_c) > 0 else 0

    @staticmethod
    def jaccard(support, support_a, support_c):
        """Jaccard = support / (support_a + support_c - support)"""
        denom = support_a + support_c - support
        return support / denom if denom > 0 else 0

    @staticmethod
    def cosine(support, support_a, support_c):
        """Cosine = support / sqrt(support_a * support_c)"""
        denom = np.sqrt(support_a * support_c)
        return support / denom if denom > 0 else 0

    @staticmethod
    def kulczynski(support, support_a, support_c):
        """
        Kulczynski = 0.5 * (support / support_a + support / support_c)
        Trả về 0 nếu denominator = 0
        """
        s1 = support / support_a if support_a > 0 else 0
        s2 = support / support_c if support_c > 0 else 0
        return 0.5 * (s1 + s2)

    def calc_metric(self, rules_df):
        rules_df = rules_df.copy()
        rules_df["leverage"] = rules_df.apply(
            lambda row: self.leverage(
                row["support"], row["support_antecedent"], row["support_consequent"]
            ),
            axis=1,
        )
        rules_df["conviction"] = rules_df.apply(
            lambda row: self.conviction(row["support_consequent"], row["confidence"]),
            axis=1,
        )
        rules_df["all_confidence"] = rules_df.apply(
            lambda row: self.all_confidence(
                row["support"], row["support_antecedent"], row["support_consequent"]
            ),
            axis=1,
        )
        rules_df["interest"] = rules_df.apply(
            lambda row: self.interest(
                row["support"], row["support_antecedent"], row["support_consequent"]
            ),
            axis=1,
        )
        rules_df["jaccard"] = rules_df.apply(
            lambda row: self.jaccard(
                row["support"], row["support_antecedent"], row["support_consequent"]
            ),
            axis=1,
        )
        rules_df["cosine"] = rules_df.apply(
            lambda row: self.cosine(
                row["support"], row["support_antecedent"], row["support_consequent"]
            ),
            axis=1,
        )
        rules_df["kulczynski"] = rules_df.apply(
            lambda row: self.kulczynski(
                row["support"], row["support_antecedent"], row["support_consequent"]
            ),
            axis=1,
        )
        return rules_df
