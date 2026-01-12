from pyspark.sql import SparkSession, DataFrame


class DataLoader:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_csv(self, path: str) -> DataFrame:
        return (
            self.spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(path)
        )
