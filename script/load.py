import duckdb
import logging

DB_PATH = r"D:\weather_etl\database\weather.duckdb"


def load_data(data: dict) -> None:
    conn = None

    try:
        conn = duckdb.connect(DB_PATH)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS weather (
                time TEXT,
                temperature DOUBLE,
                humidity INTEGER,
                wind_speed DOUBLE
            )
        """)

        exists = conn.execute(
            "SELECT COUNT(*) FROM weather WHERE time = ?", (data["time"],)
        ).fetchone()[0]

        if exists:
            logging.info("Duplicate weather data skipped")
            return

        conn.execute(
            "INSERT INTO weather VALUES (?, ?, ?, ?)",
            (
                data["time"],
                data["temperature"],
                data["humidity"],
                data["wind_speed"],
            ),
        )

        logging.info("Weather data loaded successfully")

    except duckdb.Error:
        logging.exception("Database error")
        raise

    finally:
        if conn is not None:
            conn.close()
