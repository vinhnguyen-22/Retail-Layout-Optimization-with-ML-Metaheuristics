import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


class HUIMPipeline:

    def __init__(
        self,
        df,
        spmf_jar,
        txt_input=INTERIM_DATA_DIR / "huim_input.txt",
        txt_output=PROCESSED_DATA_DIR / "huim_output.txt",
        scale_util=1e3,
    ):
        self.spmf_jar = spmf_jar
        self.scale_util = scale_util
        self.df = df
        self.item2id = None
        self.id2item = None
        self.tx_df = None
        self.txt_input = txt_input
        self.txt_output = txt_output
        self.results_df = None

    def preprocess(self):
        df = self.df.where(col("Item_id").isNotNull()).where(col("Sales").isNotNull())
        unique_items = (
            df.select("Item_id").distinct().rdd.flatMap(lambda x: x).collect()
        )
        self.item2id = {item: i + 1 for i, item in enumerate(sorted(unique_items))}
        self.id2item = {v: k for k, v in self.item2id.items()}
        from pyspark.sql.functions import udf

        # --- Sửa tại đây: dùng biến cục bộ thay cho self trong UDF
        item2id = self.item2id

        def map_item(x):
            return item2id.get(x, -1)

        map_item_udf = udf(map_item, IntegerType())
        df = df.withColumn("ItemId", map_item_udf(col("Item_id")))
        df = df.withColumn(
            "Utility", (col("Sales") / self.scale_util).cast(IntegerType())
        )
        self.df = df.toPandas()
        logger.success("✔️ Đã xong bước preprocess.")

    def filter_transactions(self, min_tx_len=3, min_percentile_utility=0.7):
        tx_df = (
            self.df.groupby("MergedId")
            .agg(
                TransactionUtility=("Utility", "sum"),
                ItemList=("ItemId", list),
                UtilList=("Utility", list),
            )
            .reset_index()
        )
        threshold = np.percentile(
            tx_df["TransactionUtility"], min_percentile_utility * 100
        )
        logger.debug(
            f"Loại các transaction utility thấp hơn {threshold:.0f}, length < {min_tx_len}"
        )
        tx_df = tx_df[
            (tx_df["TransactionUtility"] >= threshold)
            & (tx_df["ItemList"].apply(len) >= min_tx_len)
        ]
        self.tx_df = tx_df
        logger.success(f"✔️ Số transaction còn lại: {len(tx_df)}")

    def export_txt_input(self):
        with open(self.txt_input, "w", encoding="utf-8") as f:
            for _, row in self.tx_df.iterrows():
                item_str = " ".join(str(x) for x in row["ItemList"])
                util_str = " ".join(str(x) for x in row["UtilList"])
                line = f"{item_str}:{int(row['TransactionUtility'])}:{util_str}\n"
                f.write(line)
        logger.success(f"✔️ Xuất SPMF input: {self.txt_input}")

    def run(self, min_utility_threshold=None, min_len=2, max_len=3):
        total_utility = self.tx_df["TransactionUtility"].sum()
        min_utility = int(total_utility * min_utility_threshold)
        logger.warning(
            f"Chạy SPMF EFIM với min_utility={min_utility}, min_len={min_len}, max_len={max_len}..."
        )
        cmd = f'java -jar "{self.spmf_jar}" run EFIM "{self.txt_input}" "{self.txt_output}" {min_utility} {min_len} {max_len}'
        os.system(cmd)
        logger.success(f"✔️ Đã xong mining. Output: {self.txt_output}")

    def decode_results(self, min_items=2, max_items=3, top_k=100):
        results = []
        with open(self.txt_output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#UTIL:" in line:
                    item_part, util_part = line.split("#UTIL:")
                    items = item_part.strip()
                    utility = util_part.strip()
                elif ":" in line:
                    items, utility = line.split(":")
                    items = items.strip()
                    utility = utility.strip()
                else:
                    continue
                try:
                    item_names = [
                        self.id2item[int(i)]
                        for i in items.strip().split()
                        if i.isdigit()
                    ]
                    if min_items <= len(item_names) <= max_items:
                        results.append({"items": item_names, "utility": int(utility)})
                except Exception as e:
                    print(f"Lỗi parse: '{line}' - {e}")
                    continue
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="utility", ascending=False).head(top_k)
        self.results_df = results_df
        print(f"✔️ Đã giải mã {len(results_df)} itemset.")
        return results_df

    def save(self, out_csv="hui_results.csv"):
        self.results_df.to_csv(out_csv, index=False)
        logger.success(f"✔️ Đã lưu output ra: {out_csv}")
        print(self.results_df.head(10)[["items", "utility"]])
