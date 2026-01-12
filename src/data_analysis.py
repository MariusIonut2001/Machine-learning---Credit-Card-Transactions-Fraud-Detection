from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, count, hour, month, col, expr
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalysis:

    @staticmethod
    def basic_stats(df: DataFrame):
        stats = df.select("amt", "city_pop").describe().toPandas()
        print("\n=== STATISTICI DESCRIPTIVE ===")
        print(stats)

        amt_pd = df.select("amt").toPandas()
        plt.figure(figsize=(10,5))
        sns.histplot(amt_pd['amt'], bins=50, log_scale=(False, True))
        plt.title("Distributia valorilor tranzactiilor (amt)")
        plt.xlabel("Amount ($)")
        plt.ylabel("Număr tranzacții")
        plt.show()

    @staticmethod
    def fraud_distribution(df: DataFrame):
        dist = df.groupBy("is_fraud").count().toPandas()
        print("\n=== DISTRIBUTIE FRAUDA ===")
        print(dist)

        plt.figure(figsize=(6,4))
        sns.barplot(data=dist, x="is_fraud", y="count", palette="coolwarm")
        plt.title("Distributia tranzactiilor frauduloase vs normale")
        plt.xlabel("Fraudă (0=legitim, 1=fraudă)")
        plt.ylabel("Număr tranzacții")
        plt.show()

    @staticmethod
    def avg_amount_by_fraud(df: DataFrame):
        avg_df = df.groupBy("is_fraud").agg(avg("amt").alias("avg_amt")).toPandas()
        print("\n=== VALOARE MEDIE TRANZACTII ===")
        print(avg_df)

        plt.figure(figsize=(6,4))
        sns.barplot(data=avg_df, x="is_fraud", y="avg_amt", palette="Set2")
        plt.title("Valoarea medie a tranzactiilor frauduloase vs normale")
        plt.xlabel("Frauda (0=legitim, 1=frauda)")
        plt.ylabel("Valoare medie tranzactie ($)")
        plt.show()

    @staticmethod
    def fraud_by_category(df: DataFrame):
        cat_df = df.groupBy("category", "is_fraud").count().toPandas()
        print("\n=== FRAUDA PE CATEGORII ===")
        print(cat_df.head(20))

        plt.figure(figsize=(12,6))
        sns.barplot(data=cat_df, x="category", y="count", hue="is_fraud")
        plt.title("Frauda pe categorii de comercianti")
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def fraud_by_gender(df: DataFrame):
        gender_df = df.groupBy("gender", "is_fraud").count().toPandas()
        print("\n=== FRAUDA PE GEN ===")
        print(gender_df)

        plt.figure(figsize=(6,4))
        sns.barplot(data=gender_df, x="gender", y="count", hue="is_fraud")
        plt.title("Frauda pe gen")
        plt.show()

    @staticmethod
    def fraud_by_hour(df: DataFrame):
        hour_df = df.withColumn("hour", hour("trans_date_trans_time")) \
                    .groupBy("hour", "is_fraud").count().toPandas()
        print("\n=== FRAUDA PE ORE ===")
        print(hour_df.head(24))

        plt.figure(figsize=(12,5))
        sns.lineplot(data=hour_df, x="hour", y="count", hue="is_fraud", marker="o")
        plt.title("Frauda pe ore")
        plt.xlabel("Ora tranzactiei")
        plt.ylabel("Numar tranzactii")
        plt.show()

    @staticmethod
    def fraud_by_month(df: DataFrame):
        month_df = df.withColumn("month", month("trans_date_trans_time")) \
                     .groupBy("month", "is_fraud").count().toPandas()
        print("\n=== FRAUDA PE LUNI ===")
        print(month_df.head(12))

        plt.figure(figsize=(10,5))
        sns.lineplot(data=month_df, x="month", y="count", hue="is_fraud", marker="o")
        plt.title("Frauda pe luni")
        plt.xlabel("Luna")
        plt.ylabel("Numar tranzactii")
        plt.show()

    @staticmethod
    def extreme_transactions(df: DataFrame):
        extreme_df = df.filter(col("amt") > 5000) \
                       .select("amt", "category", "city", "state", "is_fraud") \
                       .orderBy(col("amt").desc()).toPandas()
        print("\n=== TRANZACTII EXTREME (amt > 5000) ===")
        print(extreme_df.head(10))

        plt.figure(figsize=(10,5))
        sns.histplot(extreme_df['amt'], bins=20, color='red')
        plt.title("Distributia tranzactiilor extreme (>5000$)")
        plt.xlabel("Amount ($)")
        plt.ylabel("Numar tranzactii")
        plt.show()

    @staticmethod
    def fraud_rate_by_category(df: DataFrame):
        fraud_stats = df.groupBy("category").agg(
            count("*").alias("total"),
            expr("sum(is_fraud)").alias("fraud_count")
        )
        fraud_stats = fraud_stats.withColumn(
            "fraud_rate_percent",
            (col("fraud_count") / col("total")) * 100
        ).orderBy(col("fraud_rate_percent").desc())

        fraud_pd = fraud_stats.toPandas()
        print("\n=== RATA FRAUDA (%) PE CATEGORII ===")
        print(fraud_pd.head(20))

        plt.figure(figsize=(12,6))
        sns.barplot(data=fraud_pd, x="category", y="fraud_rate_percent", palette="viridis")
        plt.title("Rata fraudelor (%) pe categorii")
        plt.xticks(rotation=45)
        plt.ylabel("Fraud rate (%)")
        plt.show()
