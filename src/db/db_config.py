from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

from src.config import (
    DB_DATABASE,
    DB_PASSWORD,
    DB_SERVER,
    DB_USERNAME,
    TRUSTED_CONNECTION,
)


def get_connection_string():
    if TRUSTED_CONNECTION and TRUSTED_CONNECTION.lower() == "yes":
        engine = (
            f"mssql+pyodbc://@{DB_SERVER}/{DB_DATABASE}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
            "&trusted_connection=yes"
        )
    else:
        engine = (
            f"mssql+pyodbc://{DB_USERNAME}:{DB_PASSWORD}@{DB_SERVER}/{DB_DATABASE}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
        )
    return engine


engine = create_engine(get_connection_string())


def load_query(file_path, params=None):
    """Đọc file SQL và format với params nếu cần."""
    with open(file_path, "r", encoding="utf-8-sig") as file:
        sql = file.read()
    if params:
        sql = sql.format(**params)
    return sql


def load_table(engine, sql, schema=None, postprocess=None):
    """Load table from SQL, ép kiểu schema, xử lý hậu kỳ nếu có."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
        if schema:
            schema_used = {col: typ for col, typ in schema.items() if col in df.columns}
            df = df.astype(schema_used)
        if postprocess:
            df = postprocess(df)
        return df
    except Exception as e:
        logger.error(f"Error loading table: {e}")
        raise


def execute_query(query, engine=engine):
    return pd.read_sql(query, engine)


def read_sql_parallel(path_file, query_params, engine=engine):
    with ThreadPoolExecutor() as executor:
        future_query = executor.submit(load_query, path_file, query_params)
        query = future_query.result()

        future_data = executor.submit(execute_query, query, engine)
        output = future_data.result()
    return output
