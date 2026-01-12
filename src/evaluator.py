from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:

    def __init__(self):
        self.auc_evaluator = BinaryClassificationEvaluator(
            labelCol="is_fraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

    def evaluate(self, predictions: DataFrame):
        # AUC
        auc = self.auc_evaluator.evaluate(predictions)

        pdf = predictions.select("is_fraud", "prediction").toPandas()
        y_true = pdf["is_fraud"]
        y_pred = pdf["prediction"]

        # metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc
        }

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        return metrics, cm

    @staticmethod
    def plot_confusion_matrix(cm, model_name):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    @staticmethod
    def compare_models(metrics_dict):
        # metrics_dict = {model_name: metrics}
        df = pd.DataFrame(metrics_dict).T
        df.plot(kind="bar", figsize=(10, 5))
        plt.title("Compararea modelelor")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.show()
