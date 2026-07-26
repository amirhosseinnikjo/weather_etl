import requests
import logging


def extract_data() -> dict:
    """Extract weather data from the API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 35.69,
        "longitude": 51.42,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    }
    try:
        logging.info("sending request to Open_Meto API")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logging.error("request timed out ")
        raise
    except requests.exceptions.ConnectionError:
        logging.error(" connection error")
        raise
    except requests.exceptions.HTTPError as e:
        logging.error(f"Http error:{e}")
        raise
    except requests.exceptions.RequestException as e:
        logging.exception(f"request failed:{e}")
        raise


if __name__ == "__main__":
    data = extract_data()
    print(data)
