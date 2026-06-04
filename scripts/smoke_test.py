import argparse
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from soc_ser.trainer import train_from_config


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run a tiny synthetic SOC training smoke test.")
    parser.add_argument("--work-dir", default="/tmp/soc_ser_smoke")
    args = parser.parse_args()

    if os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)

    base_dir = os.path.join(args.work_dir, "data", "toy")
    feature_dir = os.path.join(args.work_dir, "features")
    output_dir = os.path.join(args.work_dir, "outputs")
    label_map = {"Neutral": 0, "Happy": 1, "Sad": 2}
    os.makedirs(feature_dir, exist_ok=True)

    rng = np.random.default_rng(2024)
    rows = []
    for idx in range(15):
        key = f"toy-{idx:03d}"
        label = list(label_map.keys())[idx % len(label_map)]
        length = 6 + (idx % 5)
        feat = rng.normal(size=(length, 16)).astype(np.float32)
        np.save(os.path.join(feature_dir, f"{key}.npy"), feat)
        rows.append({"key": key, "emo": label})

    train_rows = rows[:12]
    test_rows = rows[12:]
    _write_jsonl(os.path.join(base_dir, "fold_1", "toy_train_fold_1.jsonl"), train_rows)
    _write_jsonl(os.path.join(base_dir, "fold_1", "toy_test_fold_1.jsonl"), test_rows)

    config = {
        "base_dir": base_dir,
        "feature_dir": feature_dir,
        "file_prefix": "toy",
        "label_map": label_map,
        "num_folds": 1,
        "input_dim": 16,
        "spd_dim": 4,
        "hidden_dim": 8,
        "device": "cpu",
        "batch_size": 4,
        "epochs": 1,
        "patience": 1,
        "lr": 1e-3,
        "output_dir": output_dir,
    }
    summary = train_from_config(config)
    print("Smoke test passed.")
    print(summary)


if __name__ == "__main__":
    main()
