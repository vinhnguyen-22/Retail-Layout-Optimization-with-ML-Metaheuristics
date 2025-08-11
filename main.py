import ast
import os
import warnings

import numpy as np
import pandas as pd
import typer

from src import fpg_analyzer, huim_analyzer, load_data, preprocess
from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

app = typer.Typer()
app.add_typer(load_data.app, name="load-data")
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(fpg_analyzer.app, name="fpgrowth")
app.add_typer(huim_analyzer.app, name="huim")
# app.add_typer(train.app, name="train")
# app.add_typer(predict.app, name="predict")
# app.add_typer(reconcile.app, name="reconcile")


@app.command()
def clean():
    # clear_files(PROCESSED_DATA_DIR, extensions=[".csv"])
    pass


if __name__ == "__main__":
    app()
