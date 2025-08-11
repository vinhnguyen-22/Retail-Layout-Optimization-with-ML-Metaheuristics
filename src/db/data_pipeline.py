# src/pipeline.py
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import RAW_DATA_DIR
from src.db.db_config import load_query, load_table


class DataPipeline:
    def __init__(self, engine, query_paths, schema, cache_dir=RAW_DATA_DIR):
        self.engine = engine
        self.query_paths = query_paths
        self.schema = schema
        self.cache_dir = Path(cache_dir)

    def _load_or_query(self, name, params=None, postprocess=None, force=False):

        cache_path = self.cache_dir / f"{name}.csv"
        schema = self.schema.get(name)

        if cache_path.exists() and not force:
            df = pd.read_csv(cache_path, encoding="utf-8-sig")
            logger.success(f"✅ Đã load cache {cache_path.name} ({len(df)} dòng)")
        else:
            sql = load_query(self.query_paths[name], params)
            df = load_table(self.engine, sql, schema, postprocess)

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            logger.success(f"💾 Đã query & cache {cache_path.name} ({len(df)} dòng)")

        return df

    def load_df(self, name, params=None, postprocess=None):
        return self._load_or_query(name, params, postprocess)

    def load_sku(self):
        return self._load_or_query("sku")

    def load_store_adjust(self):
        return self._load_or_query("store_adjust")

    def load_transactions(self, params, force_reload):
        return self._load_or_query("transactions", params=params, force=force_reload)
