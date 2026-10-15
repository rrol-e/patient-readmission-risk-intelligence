"""
Data Download Module for UCI 130-US Hospitals Diabetes Readmission Dataset
"""

import os
import zipfile
import requests
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# UCI Dataset URLs (primary and mirror fallback)
PRIMARY_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"


def download_and_extract_data(force: bool = False) -> Path:
    """
    Downloads and extracts the UCI Diabetes dataset if not already present.
    Returns path to diabetic_data.csv.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = RAW_DATA_DIR / "diabetic_data.csv"
    id_mapping_csv = RAW_DATA_DIR / "IDs_mapping.csv"

    if target_csv.exists() and id_mapping_csv.exists() and not force:
        print(f"[INFO] Raw data already present at: {RAW_DATA_DIR}")
        return target_csv

    zip_path = RAW_DATA_DIR / "dataset.zip"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    urls_to_try = [PRIMARY_URL, FALLBACK_URL]
    downloaded = False

    for url in urls_to_try:
        try:
            print(f"[INFO] Attempting to download dataset from: {url}")
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            if response.status_code == 200:
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
                downloaded = True
                print(f"[SUCCESS] Downloaded dataset archive ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
                break
            else:
                print(f"[WARN] HTTP {response.status_code} from {url}")
        except Exception as e:
            print(f"[ERROR] Failed fetching {url}: {e}")

    if not downloaded:
        raise RuntimeError("Failed to download UCI Diabetes dataset from all attempted sources.")

    # Extract contents
    print("[INFO] Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    # Some zip distributions nest another zip file inside (dataset_diabetes.zip)
    nested_zip = RAW_DATA_DIR / "dataset_diabetes.zip"
    if nested_zip.exists():
        print("[INFO] Extracting nested dataset_diabetes.zip...")
        with zipfile.ZipFile(nested_zip, "r") as nested_ref:
            nested_ref.extractall(RAW_DATA_DIR)

    # Handle if files are extracted into a subfolder (e.g. dataset_diabetes/)
    subfolder = RAW_DATA_DIR / "dataset_diabetes"
    if subfolder.exists() and subfolder.is_dir():
        for file in subfolder.iterdir():
            destination = RAW_DATA_DIR / file.name
            if not destination.exists():
                file.rename(destination)

    if not target_csv.exists():
        # Check recursively in RAW_DATA_DIR
        for found_file in RAW_DATA_DIR.rglob("diabetic_data.csv"):
            destination = RAW_DATA_DIR / "diabetic_data.csv"
            found_file.rename(destination)
            break

    if not id_mapping_csv.exists():
        for found_file in RAW_DATA_DIR.rglob("IDs_mapping.csv"):
            destination = RAW_DATA_DIR / "IDs_mapping.csv"
            found_file.rename(destination)
            break

    if not target_csv.exists():
        raise FileNotFoundError(f"diabetic_data.csv not found after extraction in {RAW_DATA_DIR}")

    print(f"[SUCCESS] Raw dataset ready: {target_csv} ({target_csv.stat().st_size / (1024*1024):.2f} MB)")
    return target_csv


if __name__ == "__main__":
    download_and_extract_data()
