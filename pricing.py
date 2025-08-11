from pathlib import Path

import numpy as np

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.dynamic_pricing_pipeline import DataLoader, DynamicPricingPipeline
from src.features.feature_engineer import FeatureEngineer
from src.models.pricing_optimize import CausalEstimator, Grouper, PricingOptimizer


def main():
    # 1. Khởi tạo các module
    data = DataLoader(
        assoc_rules_path=PROCESSED_DATA_DIR / "association_rules.csv",
        freq_itemsets_path=PROCESSED_DATA_DIR / "frequent_itemsets.csv",
        layout_real_path=INTERIM_DATA_DIR / "layout_real.csv",
        transactions_path=INTERIM_DATA_DIR / "transaction_fpg.csv",
    )
    # Sau khi load, bạn lấy assoc_rules, freq_itemsets để truyền cho FE
    feature_engineer = FeatureEngineer(data.assoc_rules, data.freq_itemsets)
    grouper = Grouper(min_sample=50, min_price_var=3)
    optimizer = PricingOptimizer(min_sample=50, min_price_var=3)
    estimator = CausalEstimator(min_price_var=3)
    # 2. Khởi tạo pipeline
    pipeline = DynamicPricingPipeline(
        data_loader=data,
        feature_engineer=feature_engineer,
        grouper=grouper,
        optimizer=optimizer,
        estimator=estimator,
        max_workers=8,  # hoặc số core bạn muốn
    )

    # 3. Chạy pipeline, xuất csv nếu muốn
    df_result = pipeline.run(export_csv=PROCESSED_DATA_DIR / "dynamic_price.csv")
    print(df_result.head())


if __name__ == "__main__":
    main()
