import duckdb
import logging


def load_data(data: dict) -> None:
    try:
        conn = duckdb.connect("Weather.duckdb")
        conn.execute("""CREATE TABLE  IF NOT EXIT weather(
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
