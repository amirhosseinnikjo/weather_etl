import logging


def transform_data(data: dict) -> dict:
    """
    Transform raw weather data.
    Args:
        data (dict): Raw weather data from the API
    Returns:
        dict: Cleaned weather data
    """
    try:
        transform_data = {
            "time": data["current"]["time"],
            "temperature": data["current"]["temperature_2m"],
            "humidity": data["current"]["relative_humidity_2m"],
            "wind_speed": data["current"]["wind_speed_10m"],
        }

        logging.info("Weather data transformed successfully.")
        return transform_data
    except KeyError as e:
        logging.exception(f"Missing required key :{e}")
        raise


if __name__ == "__main__":
    from extract import extract_data

    data = extract_data()
    df = transform_data(data)
    print(df)
