import ast

import pandas as pd
import typer
from loguru import logger
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, StringType

from src.config import INTERIM_DATA_DIR, SPMF_JAR_FILE, SparkConfig
from src.models.fpgrowth import FPGrowthRunner
from src.models.huim import HUIMPipeline
from src.spark.spark_manager import SparkSessionManager

app = typer.Typer(help="Retail Forecast Pipeline CLI")


@app.command("run_fpgrowth")
def run_fpgrowth(input_file="transaction_fpg.csv"):
    config = SparkConfig()
    with SparkSessionManager(config) as spark:
        df_transaction = spark.read.csv(
            str(INTERIM_DATA_DIR / input_file), header=True, inferSchema=True
        )
        fp_runner = FPGrowthRunner(
            transactions=df_transaction,
            min_support=0.001,
            min_confidence=0.5,
            min_lift=1.0,
            max_rules=10000,
        )
        frequent_itemsets, rules = fp_runner.run()
        frequent_itemsets_df, rules_df = fp_runner.process(
            frequent_itemsets=frequent_itemsets,
            rules=rules,
            save_dir=INTERIM_DATA_DIR,
        )


@app.command("run_huim")
def run_huim(input_file="transaction_fpg.csv"):
    config = SparkConfig()
    with SparkSessionManager(config) as spark:
        df_transaction = spark.read.csv(
            str(INTERIM_DATA_DIR / input_file), header=True, inferSchema=True
        )
        print(df_transaction.count())
        freq_df = spark.read.option("header", True).csv(
            f"{INTERIM_DATA_DIR}/frequent_itemsets.csv"
        )
        freq_df = freq_df.filter(col("support") > 0.003)

        # Tạo UDF chuyển string sang list
        def parse_items(s):
            try:
                return ast.literal_eval(s)
            except Exception:
                return []

        parse_items_udf = udf(parse_items, ArrayType(StringType()))

        # Thêm cột Items_list
        freq_df = freq_df.withColumn("Items_list", parse_items_udf(col("Items")))

        # Lấy ra tất cả các item frequent (dạng flat list hoặc set)
        flat_items = freq_df.select(explode(col("Items_list")).alias("item")).distinct()
        frequent_items = [row["item"] for row in flat_items.collect()]
        df_transaction = df_transaction.filter(col("Item_id").isin(frequent_items))
        pipeline = HUIMPipeline(
            df=df_transaction,
            spmf_jar=SPMF_JAR_FILE,
            txt_input=INTERIM_DATA_DIR / "huim_input.txt",
            txt_output=INTERIM_DATA_DIR / "huim_output.txt",
            scale_util=1e3,
        )
        pipeline.preprocess()
        pipeline.filter_transactions(min_tx_len=3, min_percentile_utility=0.8)
        pipeline.export_txt_input()
        pipeline.run(min_utility_threshold=0.02, min_len=2, max_len=5)
        pipeline.decode_results(min_items=2, max_items=5, top_k=100)
        pipeline.save(out_csv=INTERIM_DATA_DIR / "hui_results.csv")


if __name__ == "__main__":
    app()
