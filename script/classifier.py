import requests
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

LATITUDE = 35.69
LONGITUDE = 51.42

START_DATE = "2025-01-01"
END_DATE = "2026-09-01"


def get_historical_data():
    """Fetch historical weather data from Open-Meteo Archive API."""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": (
            "temperature_2m," "relative_humidity_2m," "wind_speed_10m," "weather_code"
        ),
        "timezone": "auto",
    }

    response = requests.get(
        url=url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        {
            "time": data["hourly"]["time"],
            "temperature": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "wind_speed": data["hourly"]["wind_speed_10m"],
            "weather_code": data["hourly"]["weather_code"],
        }
    )

    df["time"] = pd.to_datetime(df["time"])

    return df


def create_labels(df):
    """Convert weather codes into weather state labels."""

    df = df.copy()

    def classify_weather(code):
        """Map weather codes to weather state categories."""

        if code == 0:
            return "Clear"

        if code in [1, 2, 3]:
            return "Cloudy"

        if code in [
            51,
            53,
            55,
            56,
            57,
            61,
            63,
            65,
            66,
            67,
            80,
            81,
            82,
            95,
            96,
            99,
        ]:
            return "Rainy"

        return "Other"

    df["weather_state"] = df["weather_code"].apply(classify_weather)

    return df


def prepare_data(df):
    """Create machine learning features and target values."""

    df = df.copy()

    df["hour"] = df["time"].dt.hour

    df["day_of_week"] = df["time"].dt.dayofweek

    df["month"] = df["time"].dt.month

    df = df.dropna(
        subset=[
            "temperature",
            "humidity",
            "wind_speed",
            "weather_state",
        ]
    )

    features = [
        "temperature",
        "humidity",
        "wind_speed",
        "hour",
        "day_of_week",
        "month",
    ]

    X = df[features]
    y = df["weather_state"]

    return df, X, y


def train_model(X, y):
    """Train and evaluate a Random Forest classification model."""

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print()
    print("Training data:")
    print(y_train.value_counts())

    print()
    print("Test data:")
    print(y_test.value_counts())

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    print()
    print("=" * 50)
    print("CLASSIFICATION MODEL")
    print("=" * 50)

    print(f"Accuracy: {accuracy:.2f}")

    print()
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    return model


def predict_current_state(df, model):
    """Predict the current weather state and class probabilities."""

    df = df.copy()

    df["hour"] = df["time"].dt.hour

    df["day_of_week"] = df["time"].dt.dayofweek

    df["month"] = df["time"].dt.month

    features = [
        "temperature",
        "humidity",
        "wind_speed",
        "hour",
        "day_of_week",
        "month",
    ]

    last_row = df.iloc[-1]

    X_current = pd.DataFrame(
        [
            [
                last_row["temperature"],
                last_row["humidity"],
                last_row["wind_speed"],
                last_row["hour"],
                last_row["day_of_week"],
                last_row["month"],
            ]
        ],
        columns=features,
    )

    prediction = model.predict(X_current)[0]

    probabilities = model.predict_proba(X_current)[0]

    classes = model.classes_

    probability_df = pd.DataFrame(
        {
            "weather_state": classes,
            "probability": probabilities,
        }
    )

    probability_df["probability"] = (probability_df["probability"] * 100).round(2)

    return prediction, probability_df


def main():
    """Run the complete weather classification pipeline."""

    print("Getting historical weather data...")

    df = get_historical_data()

    print(f"Historical records: {len(df)}")

    print()
    print("Creating weather labels...")

    df = create_labels(df)

    print()
    print("Weather states:")

    print(df["weather_state"].value_counts())

    print()
    print("Preparing ML data...")

    df, X, y = prepare_data(df)

    print(f"Features shape: {X.shape}")

    print()
    print("Training Random Forest...")

    model = train_model(
        X,
        y,
    )

    print()
    print("Predicting current weather state...")

    prediction, probabilities = predict_current_state(
        df,
        model,
    )

    print()
    print("=" * 50)
    print("CURRENT WEATHER STATE")
    print("=" * 50)

    print(f"Prediction: {prediction}")

    print()
    print("Probabilities:")

    print(probabilities.to_string(index=False))


if __name__ == "__main__":
    main()
