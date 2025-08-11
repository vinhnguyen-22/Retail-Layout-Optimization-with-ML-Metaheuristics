from pyspark.sql import SparkSession


class SparkSessionManager:
    def __init__(self, config):
        self.config = config
        self.spark = None

    def __enter__(self):
        try:
            self.spark = (
                SparkSession.builder.appName("FP-Growth with PySpark")
                .master(f"local[{self.config.cores}]")
                .config("spark.driver.memory", self.config.driver_memory)
                .config("spark.executor.memory", self.config.executor_memory)
                .config(
                    "spark.sql.shuffle.partitions", str(self.config.shuffle_partitions)
                )
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel("WARN")  # <-- Giảm log spam
            print(
                f"Spark session created: appId={self.spark.sparkContext.applicationId}"
            )
            return self.spark
        except Exception as e:
            print(f"Failed to create Spark session: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.spark:
            print("Stopping Spark session...")
            self.spark.stop()
            print("Spark session stopped.")
