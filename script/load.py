import duckdb
import logging


def load_data(data: dict) -> None:
    try:
        conn = duckdb.connect("database/Weather.duckdb")
        conn.execute("""CREATE TABLE  IF NOT EXISTS weather(
                        time TEXT,
                        temperature DOUBLE,
                        humidity INTEGER,
                        wind_speed DOUBLE)""")
        conn.execute(
            """
                    INSERT INTO weather VALUES (?, ?, ?, ?)
                    """,
            (
                data["time"],
                data["temperature"],
                data["humidity"],
                data["wind_speed"],
            ),
        )
        logging.info("Weather data loaded  successfully")
        conn.close
    except duckdb.Error as e:
        logging.exception(f"Database error: {e}")
        raise
