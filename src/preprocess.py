import pandas as pd
import typer
from loguru import logger

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.utils.general_utils import convert_to_date

app = typer.Typer()


@app.command("transform")
def transform_data(
    input_file: str = "transactions.csv",
    sku_file: str = "sku.csv",
    output_file: str = "transaction_fpg.csv",
    item_id: str = "SDeptName",
):
    """Load and transform raw store data: rename columns, convert dates, normalize values."""
    df_transaction = pd.read_csv(RAW_DATA_DIR / input_file, encoding="utf-8-sig")
    df_sku = pd.read_csv(RAW_DATA_DIR / sku_file, encoding="utf-8-sig")
    df = df_transaction.merge(
        df_sku[["Sku", "Nh2", "Mfgr", "SDeptName"]], on="Sku", how="left"
    ).dropna()
    df = df[(df["Mfgr"].isin([2, 3, 4]))]
    df = df[["MergedId", item_id, "Sku", "Sales", "Quantity", "Date", "Cost"]].copy()
    df["Price"] = df["Sales"] / df["Quantity"]
    df = convert_to_date(df, "Date")
    df.rename(columns={item_id: "item_id"}, inplace=True)
    try:
        df.to_csv(
            INTERIM_DATA_DIR / output_file,
            index=False,
            date_format="%Y-%m-%d",
            encoding="utf-8-sig",
        )
        logger.success(f"Data transformed and saved to {output_file}")
        return df
    except Exception as e:
        logger.error(f"Error processing key {df['Key'][0]}: {e}")
        return None, None


if __name__ == "__main__":
    app()
