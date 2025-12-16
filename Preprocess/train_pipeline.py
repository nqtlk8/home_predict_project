"""
Run the full preprocessing + training + evaluation pipeline.
Steps:
1) Split cleaned data into train/validation.
2) Train linear regression model.
3) Evaluate model on validation set.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_required_inputs() -> None:
    cleaned_data = Path("data/cleaned/cleaned_data.csv")
    if not cleaned_data.exists():
        raise FileNotFoundError(
            "Missing cleaned data at 'data/cleaned/cleaned_data.csv'. "
            "Please run the cleaning step before this pipeline."
        )


def run_step(name: str, args: list[str]) -> None:
    print(f"=== {name} ===")
    completed = subprocess.run(args, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with code {completed.returncode}")


def main() -> None:
    ensure_required_inputs()

    python_bin = sys.executable
    steps = [
        ("Split dataset", [python_bin, "Preprocess/split_dataset.py"]),
        ("Train model", [python_bin, "Preprocess/train_model.py"]),
        ("Evaluate model", [python_bin, "Preprocess/evaluate_model.py"]),
    ]

    for name, args in steps:
        run_step(name, args)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

