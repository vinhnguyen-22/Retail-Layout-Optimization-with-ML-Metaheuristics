import os

import hydra
import pandas as pd
import typer
from loguru import logger

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, SparkConfig
from src.models.fpgrowth import FPGrowthRunner
from src.spark_manager import SparkSessionManager

app = typer.Typer(help="Retail Forecast Pipeline CLI")


@app.command("run")
def main(input_file="transaction_fpg.csv"):
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
            save_dir=PROCESSED_DATA_DIR,
        )


if __name__ == "__main__":
    app()
