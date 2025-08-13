import random
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
import typer
from loguru import logger

from src.config import INTERIM_DATA_DIR
from src.pipelines.layout_opt_pipeline import LayoutOptimizationPipeline
from src.preprocess import DataLoader

app = typer.Typer(help="Retail Forecast Pipeline CLI")


@app.command("run")
def run(
    assoc_rules_path: str = str("association_rules.csv"),
    freq_itemsets_path: str = str("frequent_itemsets.csv"),
    layout_real_path: str = str("layout.csv"),
    margin_matrix_path: str = None,
    n_trials: int = 30,
    n_gen_final: int = 100,
    selection: str = "tournament",
    crossover: str = "PMX",
    mutation: str = "shuffle",
    adaptive: bool = True,
    seed: int = 42,
):
    data = DataLoader(
        assoc_rules_path=INTERIM_DATA_DIR / assoc_rules_path,
        freq_itemsets_path=INTERIM_DATA_DIR / freq_itemsets_path,
        layout_real_path=INTERIM_DATA_DIR / layout_real_path,
        margin_matrix_path=margin_matrix_path,
    )

    pipeline = LayoutOptimizationPipeline(
        data=data,
        n_trials=n_trials,
        n_gen_final=n_gen_final,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        adaptive=adaptive,
        seed=seed,
        pop_size=500,
    )

    pipeline.tune()
    pipeline.run_final()
    pipeline.plot_all()


if __name__ == "__main__":
    app()
