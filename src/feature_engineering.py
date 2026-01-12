from pyspark.sql import DataFrame
from pyspark.sql.functions import hour, month
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler
)
from pyspark.ml import Pipeline


class FeatureEngineering:
    @staticmethod
    def prepare(df: DataFrame) -> DataFrame:
        # selectare coloane de interes
        df = df.select(
            "amt",
            "category",
            "gender",
            "city_pop",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "trans_date_trans_time",
            "is_fraud"
        ).dropna()

        df = df.withColumn(
            "hour", hour("trans_date_trans_time")
        ).withColumn(
            "month", month("trans_date_trans_time")
        )

        category_indexer = StringIndexer(
            inputCol="category",
            outputCol="category_idx"
        )

        gender_indexer = StringIndexer(
            inputCol="gender",
            outputCol="gender_idx"
        )

        feature_cols = [
            "amt",
            "city_pop",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "hour",
            "month",
            "category_idx",
            "gender_idx"
        ]

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )

        pipeline = Pipeline(stages=[
            category_indexer,
            gender_indexer,
            assembler
        ])

        model = pipeline.fit(df)
        final_df = model.transform(df)

        return final_df.select("features", "is_fraud")
