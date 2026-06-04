import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from soc_ser.trainer import train_from_config
from soc_ser.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train SOC for speech emotion recognition.")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = train_from_config(config)
    avg = summary["average"]
    print(f"Average: WA={avg['WA']:.4f}, UA={avg['UA']:.4f}, F1={avg['F1']:.4f}")


if __name__ == "__main__":
    main()

