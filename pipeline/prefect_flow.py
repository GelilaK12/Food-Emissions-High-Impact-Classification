from prefect import flow, task
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@task
def run_etl():
    subprocess.run(
        [sys.executable, ROOT / "pipeline" / "etl.py"],
        check=True
    )

@task
def run_validation():
    subprocess.run(
        [sys.executable, ROOT / "pipeline" / "validation.py"],
        check=True
    )

@flow(name="food_emissions_classification_pipeline")
def food_emissions_pipeline():
    run_etl()
    run_validation()

if __name__ == "__main__":
    food_emissions_pipeline()
