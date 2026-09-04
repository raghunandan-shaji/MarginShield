from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


FIFAR_URL = "https://ndownloader.figshare.com/files/52147616"
FIFAR_MD5 = "2255e7b8a391d0ab567f37ec27e52f3d"
FIFAR_MEMBERS = [
    "FiFAR/alert_data/processed_data/alerts.parquet",
    "FiFAR/alert_data/processed_data/BAF_alert_model_score.parquet",
    "FiFAR/synthetic_experts/prob_of_error.parquet",
    "FiFAR/synthetic_experts/expert_parameters.parquet",
    "FiFAR/synthetic_experts/expert_predictions.parquet",
    "FiFAR/synthetic_experts/expert_ids.yaml",
]


def run_kaggle(dataset: str, destination: Path, filename: str | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(destination), "--unzip"]
    if filename:
        command.extend(["-f", filename])
    subprocess.run(command, check=True)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_fifar(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "FiFAR.zip"
    with urllib.request.urlopen(FIFAR_URL) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    if md5(archive) != FIFAR_MD5:
        raise ValueError("FiFAR checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        for member in FIFAR_MEMBERS:
            bundle.extract(member, destination / "extracted")


def extract_baf(destination: Path) -> None:
    archive = destination / "Base.csv"
    extracted = destination / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
    elif archive.exists():
        shutil.copy2(archive, extracted / "Base.csv")
    else:
        raise FileNotFoundError("Kaggle did not produce the BAF Base.csv file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the four source datasets used by MarginShield v0.1")
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    run_kaggle("olistbr/brazilian-ecommerce", destination / "olist")
    run_kaggle("vbinh002/fraud-ecommerce", destination / "fraudecom")
    run_kaggle("sgpjesus/bank-account-fraud-dataset-neurips-2022", destination / "baf", "Base.csv")
    extract_baf(destination / "baf")
    fetch_fifar(destination / "fifar")
    print(f"Sources ready at {destination}")


if __name__ == "__main__":
    main()
