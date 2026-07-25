import os
import time

import pandas as pd
from requests import get

TITANIC_PATH = os.path.join("datasets", "titanic")
os.makedirs(TITANIC_PATH, exist_ok=True)

DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/"
)


def fetch_titanic_data(
    url=DOWNLOAD_URL, path=TITANIC_PATH, max_retries=3, retry_delay=5
):
    for attempt in range(max_retries):
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
            for filename in ("train.csv", "test.csv"):
                filepath = os.path.join(path, filename)
                if not os.path.isfile(filepath):
                    print(
                        f"Downloading {filename} (attempt {attempt + 1}/{max_retries})"
                    )
                    response = get(url + filename)
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(response.content)
            return  # Success, no need to retry
        except Exception as e:
            print(f"An error occurred (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    print("Failed to download data after multiple retries.")


def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)
