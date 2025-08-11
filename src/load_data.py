import os

import pandas as pd
import typer
from loguru import logger
from sqlalchemy import create_engine

from src.config import QUERY_PATHS
from src.db.data_pipeline import DataPipeline
from src.db.db_config import get_connection_string
from src.db.schema import SCHEMA

app = typer.Typer(help="Retail Forecast Pipeline CLI")


@app.command("run")
def run(
    date: int = typer.Option(250630, help="Ngày kết thúc (VD: 250531)"),
    store_id: int = typer.Option(112, help="StoreId (VD: 112)"),
    force_reload: bool = typer.Option(False, help="Bỏ qua cache và query lại toàn bộ"),
):
    engine = create_engine(get_connection_string())
    pipeline = DataPipeline(engine, QUERY_PATHS, SCHEMA)
    store_adjust = pipeline.load_store_adjust()
    sku = pipeline.load_sku()
    transactions = pipeline.load_transactions(
        {"date": date, "store_id": store_id}, force_reload=True
    )

    logger.success("🎉 Hoàn tất xử lý dữ liệu!")
    return store_adjust, sku, transactions


if __name__ == "__main__":
    app()
