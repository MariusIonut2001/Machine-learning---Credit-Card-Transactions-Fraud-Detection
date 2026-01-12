from src.spark_manager import SparkManager
from src.data_loader import DataLoader
from src.data_analysis import DataAnalysis
from src.feature_engineering import FeatureEngineering
from src.models import MLModels
from src.evaluator import ModelEvaluator
from src.logger import Logger

def main():
    logger = Logger.get_logger("MAIN")
    logger.info("Pornire aplicatie Big Data Fraud Detection")

    spark = SparkManager.get_spark_session()
    loader = DataLoader(spark)
    df = loader.load_csv("data/fraudTest.csv")

    # Analiza exploratorie
    DataAnalysis.basic_stats(df)
    DataAnalysis.fraud_distribution(df)
    DataAnalysis.avg_amount_by_fraud(df)
    DataAnalysis.fraud_by_category(df)
    DataAnalysis.fraud_by_gender(df)
    DataAnalysis.fraud_by_hour(df)
    DataAnalysis.fraud_by_month(df)
    DataAnalysis.extreme_transactions(df)
    DataAnalysis.fraud_rate_by_category(df)

    data = FeatureEngineering.prepare(df)
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    # Modele
    models = {
        "Logistic Regression": MLModels.logistic_regression(),
        "Decision Tree": MLModels.decision_tree(),
        "Random Forest": MLModels.random_forest()
    }

    evaluator = ModelEvaluator()
    all_metrics = {}

    logger.info("Pornire antrenare modele si evaluare...")
    for name, model in models.items():
        logger.info(f"Antrenare {name}...")
        trained = model.fit(train)
        predictions = trained.transform(test)

        metrics, cm = evaluator.evaluate(predictions)
        all_metrics[name] = metrics

        logger.info(f"\n=== {name} ===")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

        evaluator.plot_confusion_matrix(cm, name)

    evaluator.compare_models(all_metrics)

    spark.stop()
    logger.info("Aplicatie finalizata cu succes")

if __name__ == "__main__":
    main()
