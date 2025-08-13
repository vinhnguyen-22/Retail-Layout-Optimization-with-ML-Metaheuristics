import ast
import datetime
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm


def array_to_list(x):
    if isinstance(x, np.ndarray):
        return list(x)
    elif isinstance(x, list):
        return x
    elif isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [x]
    else:
        return [str(x)]


def convert_to_date(df: pd.DataFrame, cols):
    """
    Converts one or more columns from YYMMDD int format to datetime.
    Supports both single column name (str) and list of column names.
    """
    df = df.copy()
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:

        def parse_ymd(x):
            try:
                if pd.isnull(x):
                    return pd.NaT
                x = int(x)
                return datetime.date((x // 10000) + 2000, (x % 10000) // 100, x % 100)
            except Exception as e:
                print(f"[Warning] Invalid value in column '{col}': {x} → {e}")
                return pd.NaT

        df[col] = df[col].apply(parse_ymd)
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def safe_to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [x]
    return [x]


def sanitize_key(name: str) -> str:
    """Chuyển tên nhóm thành tên file an toàn."""
    key = unicodedata.normalize("NFD", name)
    key = "".join([c for c in key if unicodedata.category(c) != "Mn"])
    key = key.lower()
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", key)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized


def clear_files(directory: Path, extensions: list, delete_folders: bool = True):
    if not directory.exists():
        typer.echo(f"Directory {directory} does not exist, skip.")
        return
    files = [
        f
        for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    ]
    for file in tqdm(files, desc=f"Deleting files in {directory}"):
        file.unlink()

    if delete_folders:
        folders = [f for f in directory.iterdir() if f.is_dir()]
        for folder in tqdm(folders, desc=f"Deleting subfolders in {directory}"):
            shutil.rmtree(folder)
