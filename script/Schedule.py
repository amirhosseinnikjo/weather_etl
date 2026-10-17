import schedule
import time
import logging
import duckdb

from main import main

from classifier import get_historical_data
from classifier import create_labels
from classifier import prepare_data
from classifier import train_model
from classifier import predict_current_state

from random_farest import load_data
from random_farest import add_features
from random_farest import train_models
from random_farest import predict_next

DB_PATH = r"D:\weather_etl\database\weather.duckdb"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_prediction(predictions, weather_state):
    """Save ML prediction to DuckDB."""

    conn = duckdb.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            created_at TIMESTAMP,
            predicted_temperature DOUBLE,
            predicted_humidity DOUBLE,
            predicted_wind_speed DOUBLE,
            weather_state TEXT
        )
    """)

    conn.execute(
        """
        INSERT INTO predictions VALUES (
            CURRENT_TIMESTAMP,
            ?,
            ?,
            ?,
            ?
        )
        """,
        [
            predictions["temperature_next"],
            predictions["humidity_next"],
            predictions["wind_speed_next"],
            weather_state,
        ],
    )

    conn.close()


def run_etl():
    """Run ETL and machine learning pipeline."""

    logging.info("Starting scheduled pipeline...")

    try:

        main()

        logging.info("ETL completed successfully")

        print()
        print("===== REGRESSION =====")

        df = load_data()

        df = add_features(df)

        models = train_models(df)

        predictions = predict_next(df, models)

        print(f"Predicted temperature: " f"{predictions['temperature_next']:.2f}")

        print(f"Predicted humidity: " f"{predictions['humidity_next']:.2f}")

        print(f"Predicted wind speed: " f"{predictions['wind_speed_next']:.2f}")

        print()
        print("===== CLASSIFICATION =====")

        classifier_df = get_historical_data()

        classifier_df = create_labels(classifier_df)

        classifier_df, X, y = prepare_data(classifier_df)

        classifier_model = train_model(X, y)

        prediction, probabilities = predict_current_state(
            classifier_df, classifier_model
        )

        print(f"Weather state: {prediction}")

        print()
        print(probabilities.to_string(index=False))

        save_prediction(predictions, prediction)

        logging.info("Prediction saved successfully")

        logging.info("ETL and ML completed successfully")

    except Exception:
        logging.exception("Pipeline failed")


run_etl()

schedule.every(60).minutes.do(run_etl)


while True:

    schedule.run_pending()

    time.sleep(1)
