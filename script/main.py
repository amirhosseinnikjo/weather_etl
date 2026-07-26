from extract import extract_data
from transform import transform_data
from load import load_data
import logging


def main() -> None:
    data = extract_data()
    transformed_data = transform_data(data)
    load_data(transformed_data)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
