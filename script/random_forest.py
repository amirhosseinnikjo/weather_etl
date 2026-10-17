import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DB_PATH = r"D:\weather_etl\database\weather.duckdb"
START_DATE = "2026-07-01 00:00:00"
END_DATE = "2026-10-11 23:59:59"
TRAIN_FRACTION = 0.8

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "temperature_lag1",
    "humidity_lag1",
    "wind_speed_lag1",
]

TARGET_COLS = [
    "temperature_next",
    "humidity_next",
    "wind_speed_next",
]


def load_data() -> pd.DataFrame:
    """Load weather data from DuckDB and filter the historical date range."""
    conn = duckdb.connect(DB_PATH)

    df = conn.execute("SELECT * FROM weather").fetch_df()

    conn.close()

    df["time"] = pd.to_datetime(df["time"])

    df = df[(df["time"] >= START_DATE) & (df["time"] <= END_DATE)]

    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")
    print(df["time"].diff().value_counts().head(10))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create valid one-hour lag and next-step features."""
    df = df.copy()
    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    one_hour = pd.Timedelta(hours=1)
    previous_is_hourly = df["time"].diff().eq(one_hour)
    next_is_hourly = (df["time"].shift(-1) - df["time"]).eq(one_hour)

    for col in ["temperature", "humidity", "wind_speed"]:
        df[f"{col}_lag1"] = df[col].shift(1).where(previous_is_hourly)
        df[f"{col}_next"] = df[col].shift(-1).where(next_is_hourly)

    return df


def train_models(
    df: pd.DataFrame,
) -> dict[str, RandomForestRegressor]:
    """Train and evaluate one model per weather target."""
    data = df.dropna(subset=FEATURE_COLS + TARGET_COLS).copy()

    if len(data) < 20:
        raise ValueError("Not enough valid hourly data to train models.")

    split_index = int(len(data) * TRAIN_FRACTION)
    train_df = data.iloc[:split_index]
    test_df = data.iloc[split_index:]

    if train_df.empty or test_df.empty:
        raise ValueError("Not enough data for training and testing.")

    models = {}

    for target in TARGET_COLS:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )

        model.fit(train_df[FEATURE_COLS], train_df[target])
        models[target] = model

        y_pred = model.predict(test_df[FEATURE_COLS])
        mae = mean_absolute_error(test_df[target], y_pred)
        r2 = r2_score(test_df[target], y_pred)

        print(f"Target: {target}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.2f}")
        print()

    return models


def predict_next(
    df: pd.DataFrame,
    models: dict[str, RandomForestRegressor],
) -> dict[str, float]:
    """Predict the next hourly weather values."""
    valid = df.dropna(subset=FEATURE_COLS)

    if valid.empty:
        raise ValueError("No valid recent hourly row. Check the latest timestamps.")

    last_row = valid.iloc[-1]
    x_new = last_row[FEATURE_COLS].to_frame().T

    predictions = {}

    for target, model in models.items():
        predictions[target] = float(model.predict(x_new)[0])

    return predictions


def main() -> None:
    """Run the complete weather machine learning pipeline."""
    df = load_data()
    df = add_features(df)

    models = train_models(df)

    predictions = predict_next(df, models)

    print("\n===== Next Prediction =====")
    print(f"Predicted temperature: " f"{predictions['temperature_next']:.2f}")
    print(f"Predicted humidity: " f"{predictions['humidity_next']:.2f}")
    print(f"Predicted wind speed: " f"{predictions['wind_speed_next']:.2f}")


if __name__ == "__main__":
    main()
