# 5. System Architecture

## Modeling & Feature Engineering Toolkit

| **Tool / Library**       | **Purpose & Highlights**                                                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Scikit-learn**         | Essential Python library for machine learning, supporting a wide range of supervised and unsupervised algorithms, as well as data preprocessing and model evaluation.          |
| **Optuna**               | Automated hyperparameter optimization framework; efficiently searches for optimal configurations through adaptive and distributed experimentation.                             |
| **TSFresh**              | Automates the extraction of a large number of time series features, enabling advanced signal processing and feature engineering for temporal data.                             |
| **SHAP**                 | Delivers model interpretability via Shapley values; quantifies the contribution of each feature to individual predictions for robust explainability and auditing.              |
| **HierarchicalForecast** | Specialized library for forecasting hierarchical time series, ensuring coherence and consistency across all levels of data aggregation (e.g., national, regional, store, SKU). |

---

## MLOps, DataOps & Productionization

- **DVC**: Data and pipeline versioning for full reproducibility and collaboration
- **MLflow**: Experiment and model tracking, registry, and deployment
- **Apache Airflow / Dagster**: Workflow orchestration for ETL, training, evaluation, and reporting pipelines
- **FastAPI**: High-performance, production-ready API serving for real-time inference
- **Streamlit**: Business Intelligence dashboards for interactive analytics and visual monitoring
- **Optuna**: Integrated with pipelines for scalable, automated hyperparameter search
- **Slack (or alert system)**: Notifications, anomaly detection, and workflow monitoring

| **Tool / Component** | **Key Role in Production Workflow**                                                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Streamlit**        | Interactive dashboards for operational monitoring, real-time visualization of trends, model outputs, and error analysis. Connects live to APIs or batch outputs.      |
| **DVC**              | Data and pipeline version control; tracks every change in datasets, model artifacts, and intermediate files. Enables reproducibility and seamless team collaboration. |
| **MLflow**           | End-to-end ML lifecycle management. Tracks experiments, metrics, hyperparameters, and model versions; supports model registry and REST API/batch serving.             |
| **Dagster**          | Modern orchestrator for building, scheduling, and monitoring complex data and ML workflows. Ensures reliability and observability in production pipelines.            |
| **Apache Airflow**   | Industry-standard DAG-based orchestrator. Automates ETL, model training, evaluation, and scheduled inference. Enables robust error handling and pipeline reusability. |

---

**This technology stack ensures high scalability, reliability, and full traceability for all machine learning and data operations—from data ingestion and feature engineering to model deployment and business reporting.**
