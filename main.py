import ast
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import typer

from src import association_rules, load_data, preprocess, run_optimize_layout
from src.config import (
    FEATURE_STORE_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MLFLOW_DIR,
    MODELS_DIR,
    OUTPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from src.utils.general_utils import clear_files

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

app = typer.Typer()
app.add_typer(load_data.app, name="load-data")
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(association_rules.app, name="association_rules")
app.add_typer(run_optimize_layout.app, name="layout-opt")
# app.add_typer(train.app, name="train")
# app.add_typer(predict.app, name="predict")
# app.add_typer(reconcile.app, name="reconcile")


@app.command()
def clean():
    clear_files(PROCESSED_DATA_DIR, extensions=[".csv"])
    clear_files(
        OUTPUT_DATA_DIR,
        extensions=[".csv"],
    )
    clear_files(
        INTERIM_DATA_DIR,
        extensions=[".csv"],
    )
    clear_files(
        FIGURES_DIR,
        extensions=[".png", ".jpg", ".jpeg", ".svg"],
        delete_folders=True,
    )
    clear_files(
        MODELS_DIR,
        extensions=[".pkl"],
    )
    clear_files(
        FEATURE_STORE_DIR,
        extensions=[".json", ".parquet"],
    )
    if MLFLOW_DIR.exists():
        shutil.rmtree(MLFLOW_DIR)


if __name__ == "__main__":
    app()
