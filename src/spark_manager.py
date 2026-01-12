from pyspark.sql import SparkSession


class SparkManager:
    @staticmethod
    def get_spark_session():
        return (
            SparkSession.builder
            .appName("FraudDetectionBigData")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
