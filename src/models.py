from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)


class MLModels:
    @staticmethod
    def logistic_regression():
        return LogisticRegression(
            labelCol="is_fraud",
            featuresCol="features"
        )

    @staticmethod
    def decision_tree():
        return DecisionTreeClassifier(
            labelCol="is_fraud",
            featuresCol="features"
        )

    @staticmethod
    def random_forest():
        return RandomForestClassifier(
            labelCol="is_fraud",
            featuresCol="features",
            numTrees=50
        )
