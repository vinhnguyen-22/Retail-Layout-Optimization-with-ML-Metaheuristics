import ast
from typing import Optional

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


# =============== Data Loader ===============
class DataLoader:
    def __init__(
        self,
        assoc_rules_path,
        freq_itemsets_path,
        layout_real_path,
        margin_matrix_path: Optional[str] = None,
    ):
        self.assoc_rules = pd.read_csv(assoc_rules_path)
        self.freq_itemsets = pd.read_csv(freq_itemsets_path)
        self.layout_real = pd.read_csv(layout_real_path).drop_duplicates(keep="first")
        self.margin_matrix = (
            pd.read_csv(margin_matrix_path, index_col=0)
            if margin_matrix_path is not None
            else None
        )
        self._process()

    def _process(self):
        antecedents = self.assoc_rules["antecedent"].apply(ast.literal_eval)
        consequents = self.assoc_rules["consequent"].apply(ast.literal_eval)
        itemsets = self.freq_itemsets["items"].apply(ast.literal_eval)

        all_items = set()
        for ser in antecedents.tolist() + consequents.tolist():
            all_items.update(ser)
        for sublist in itemsets.tolist():
            all_items.update(sublist)

        layout_cats = self.layout_real["Category"].dropna().astype(str).tolist()
        all_items.update(layout_cats)
        self.all_items: List[str] = sorted(all_items)

        if {"x", "y"}.issubset(self.layout_real.columns):
            self.positions = list(zip(self.layout_real["x"], self.layout_real["y"]))
        else:
            raise ValueError("layout_real.csv thiếu cột x,y")

        # fill defaults
        for col in ["is_refrigerated", "is_entrance", "is_cashier", "width", "height"]:
            if col not in self.layout_real.columns:
                self.layout_real[col] = 0

        self.refrig_cats = (
            self.layout_real.loc[self.layout_real["is_refrigerated"] == 1, "Category"]
            .astype(str)
            .tolist()
        )

    # helpers
    def sorted_slots_xy(self) -> pd.DataFrame:
        df = self.layout_real.sort_values(["y", "x"]).reset_index(drop=True)
        if "width" not in df.columns:
            df["width"] = 0
        if "height" not in df.columns:
            df["height"] = 0
        return df[["Category", "x", "y", "width", "height"]].copy()


if __name__ == "__main__":
    app()
