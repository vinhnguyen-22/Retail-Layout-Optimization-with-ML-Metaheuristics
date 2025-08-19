# Retail Layout Optimization with ML & Metaheuristics

> Data-driven optimization for in-store product placement and floor planning using association analysis & evolutionary search.

This repository provides an end-to-end pipeline to **analyze retail transactions**, **learn affinities and association rules**, and **search for near-optimal layouts** using **metaheuristics** (Genetic Algorithm, Simulated Annealing, Tabu Search).  
The objective is to **increase sales**, **improve customer experience**, and **maximize retail space utilization**.

---

## 🔑 Key Features

- **From Data to Layouts**
  - ETL & preprocessing for retail data (transactions, SKUs, store constraints)
  - Association mining (frequent itemsets, rules, HUIM)
  - Affinity graph & layout scoring with real-world constraints
- **Metaheuristic Optimization**
  - GA (via [DEAP](https://github.com/DEAP/deap)), Simulated Annealing, Tabu Search with modular operators & callbacks
  - Objective combines **revenue/margin uplift**, **cross-sell affinity**, and **constraint penalties**
- **Experimentation & Visualization**
  - Jupyter notebooks for EDA and benchmarking
  - Reports (plots & HTML) stored under `reports/figures`
- **MLOps-Friendly**
  - `dvc.yaml` to version data and reproduce pipelines
  - Documentation in `docs/` (build with MkDocs)

---

## 🏗️ Project Structure

```
Retail-Layout-Optimization-with-ML-Metaheuristics/
├── data/                       # Data (input & artifacts)
│   ├── raw/                    # Raw input data (transactional logs, SKUs, store info)
│   │   ├── sku.csv
│   │   ├── store_adjust.csv
│   │   └── transactions.csv
│   ├── interim/                # Intermediate results from preprocessing/mining
│   │   ├── association_rules.csv
│   │   ├── frequent_itemsets.csv
│   │   ├── hui_results.csv
│   │   ├── layout.csv
│   │   └── transaction_fpg.csv
│   ├── processed/              # Final processed data ready for experiments
│   │   └── ga_logbook_final.csv
│   └── output/                 # Outputs generated from pipelines (final layouts, metrics)
│
├── docs/                       # Documentation (MkDocs site)
│   ├── index.md                # Docs homepage
│   ├── install.md              # Installation guide
│   ├── algorithms.md           # Algorithm details
│   ├── data.md                 # Data schema & structure
│   ├── architecture.md         # Project architecture
│   ├── usage.md                # CLI usage & configs
│   └── mkdocs.yml              # MkDocs configuration
│
├── models/                     # Saved ML/heuristic models (checkpoints, weights)
│
├── notebooks/                  # Jupyter notebooks for exploration & prototyping
│   ├── fpgrowth.ipynb          # Mining frequent patterns
│   ├── ga.ipynb                # Genetic Algorithm optimization demo
│   └── optimize_layout_sa.ipynb # Simulated Annealing variant
│
├── reports/                    # Reports & results
│   └── figures/                # Visualizations & plots
│       ├── affinity_heatmap.png
│       ├── ga_convergence.png
│       ├── ga_compare.png
│       └── layout_preview.html
│
├── src/                        # Main source code
│   ├── db/                     # Database schema & data pipeline
│   │   ├── data_pipeline.py
│   │   └── schema.py
│   ├── features/               # Feature engineering
│   │   └── feature_engineer.py
│   ├── models/                 # Core models & optimization algorithms
│   │   ├── ga/                 # GA operators/implementation
│   │   ├── affinity.py         # Affinity score computation
│   │   ├── fpgrowth.py         # Frequent pattern mining
│   │   ├── ga_optimizer.py     # GA optimization logic
│   │   ├── greedy.py           # Greedy baseline
│   │   └── huim.py             # High Utility Itemset Mining
│   ├── pipelines/              # Orchestration pipelines
│   │   └── layout_opt_pipeline.py
│   ├── services/               # Business/domain services
│   │   ├── affinity_services.py
│   │   ├── layout_context.py
│   │   └── layout_tuner.py
│   ├── utils/                  # Utility functions
│   │   ├── general_utils.py
│   │   ├── plot_utils.py
│   │   └── rule_metrics.py
│   ├── config.py               # Central configuration file
│   ├── preprocess.py           # Data preprocessing
│   └── run_optimize_layout.py  # Main entrypoint for optimization pipeline
│
├── app.py                      # Optional: Flask/FastAPI app or demo entrypoint
├── main.py                     # Main script (could wrap pipeline execution)
├── Makefile                    # Common commands (e.g., `make run`, `make docs`)
├── pyproject.toml              # Project metadata & dependencies (Poetry/pip-tools)
├── requirements.txt            # Python dependencies
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC lock file
└── README.md                   # Project overview (main documentation entrypoint)

```

---

---

## ⚙️ Installation

```bash
# 1) Clone repository
git clone https://github.com/vinhnguyen-22/Retail-Layout-Optimization-with-ML-Metaheuristics.git
cd Retail-Layout-Optimization-with-ML-Metaheuristics

# 2) Create virtual environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

## Pipeline Usage

```python

from enum import Enum
import typer
from src.config import INTERIM_DATA_DIR
from src.pipelines.layout_opt_pipeline import LayoutOptimizationPipeline
from src.preprocess import DataLoader

data = DataLoader(
    assoc_rules_path=INTERIM_DATA_DIR / assoc_rules_path,
    freq_itemsets_path=INTERIM_DATA_DIR / freq_itemsets_path,
    layout_real_path=INTERIM_DATA_DIR / layout_real_path,
    margin_matrix_path=margin_matrix_path,
)

pipeline = LayoutOptimizationPipeline(
    data=data,
    n_trials=n_trials,
    n_gen_final=n_gen_final,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    adaptive=adaptive,
    seed=seed,
    pop_size=500,
)

pipeline.tune()
pipeline.run_final()
pipeline.plot_all()
```

---

## 🗃️ Data Requirements

Expected files under `data/`:

- `data/raw/transactions.csv` – transaction logs (e.g., `transaction_id, sku, qty, price, ts, store_id`)
- `data/raw/sku.csv` – product master data (e.g., `sku, category, subcategory, size, brand, margin`)
- `data/raw/store_adjust.csv` – store constraints (e.g., adjacency rules, shelf capacity, forbidden zones)
- `data/interim/` & `data/processed/` – artifacts generated during mining and preprocessing (frequent itemsets, association rules, HUIM results)

> See schema definitions in `src/db/schema.py` and corresponding data pipeline scripts.

---

## 🚀 Quickstart

### 1) Preprocessing & Association Mining

```bash
# (Option A) Run script
python main.py association_rules run_fpgrowth --input-file transaction_fpg.csv

# (Option B) DVC
dvc fpgrowth
```

Outputs will be stored in `data/interim/` & `data/processed/`.

### 2) Layout Optimization

```bash
python -m src.run_optimize_layout \
  --n-trials 30 \
  --n-gen-final 200 \
  --selection "tournament" \
  --crossover "cxTwoPoint" \
  --mutation "mutShuffleIndexes" \
  --adaptive true \
  --seed 42 \
  --pop-size 500
```

- **Inputs**: mining results (`association_rules.csv`, `frequent_itemsets.csv`, `layout.csv`, etc.)
- **Outputs**: optimized layouts, GA/SA logs, and visualizations under `reports/figures/` (e.g., `ga_convergence.png`, `ga_compare.png`)

---

## 🧠 Algorithms

- **Affinity & Rule Mining**

  - Frequent Itemsets, Association Rules, High Utility Itemset Mining (HUIM)

- **Objective Function**

  - maximize **expected revenue/margin**
  - maximize **affinity adjacency** (place related items close together)
  - minimize **penalties** (constraint violations, shelf capacity, customer flow)

- **Metaheuristics**

  - **Genetic Algorithm (DEAP)** – selection, crossover, mutation, elitism
  - **Simulated Annealing** – temperature schedule & neighborhood search
  - **Tabu Search** – tabu list, aspiration, diversification

---

## 🧪 Reproducibility & Tracking

- **DVC**

  ```bash
  dvc repro   # reproduce pipeline
  dvc push    # push artifacts to remote
  ```

- **Reports**

  - Generated figures & previews under `reports/figures/` (e.g., `ga_preview.html`)

---

## 📊 Visualization

- Heatmaps, affinity graphs
- GA/SA convergence plots
- Final layout previews
  All stored under `reports/figures/`.

---

## ⚙️ Configuration

- Modify CLI arguments:

  - `--n-trials`, `--n-gen-final`, `--pop-size`
  - `--selection`, `--crossover`, `--mutation`
  - `--adaptive`, `--seed`

- Extend GA operators in `src/models/ga/`.

---

## 🗺️ Roadmap

- Multi-objective optimization (Pareto frontier: revenue vs. walking distance)
- More realistic retail constraints (planogram rules, shelf facings, aisle flows)
- Spark integration for large-scale mining
- Additional heuristics/algorithms

---

## 🤝 Contributing

Contributions are welcome! Please open issues or PRs with improvements, new features, or bug fixes.

---

## 📄 License

MIT License (see `LICENSE` file if available).

---

## 📚 References

- [DEAP: Distributed Evolutionary Algorithms in Python](https://github.com/DEAP/deap)
- [Retail Store Layout Optimization for Maximum Product Visibility (arXiv)](https://arxiv.org/abs/2105.09299)
