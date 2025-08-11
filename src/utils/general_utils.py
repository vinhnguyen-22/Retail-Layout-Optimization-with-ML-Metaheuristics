import ast
import datetime

import numpy as np
import pandas as pd


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


def extract_all_items(series):
    items = []
    for ser in series:
        items += [item for sublist in ser.apply(eval) for item in sublist]
    return items


import pandas as pd
