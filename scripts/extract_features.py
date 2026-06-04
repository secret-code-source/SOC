import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

from soc_ser.utils import load_config, resolve_device


TARGET_SAMPLE_RATE = 16000


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_audio(path, channel=1):
    wav, sr = torchaudio.load(path)
    wav = wav[channel - 1]
    if sr != TARGET_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SAMPLE_RATE)
    return wav


def extract_features(config):
    source = config["backbone_source"]
    input_jsonl = config["input_jsonl"]
    output_dir = config["feature_dir"]
    device = resolve_device(config.get("device", "auto"))
    output_norm = bool(config.get("output_norm", True))
    dtype = config.get("dtype", "float16")

    os.makedirs(output_dir, exist_ok=True)
    processor = AutoFeatureExtractor.from_pretrained(source)
    model = AutoModel.from_pretrained(source).to(device)
    model.eval()

    for item in tqdm(list(_iter_jsonl(input_jsonl)), desc="Extracting"):
        key = item.get("key") or item.get("sid") or item.get("id")
        wav_path = item.get("wav") or item.get("audio")
        if key is None or wav_path is None:
            continue
        if not os.path.exists(wav_path):
            continue

        channel = int(item.get("channel", 1))
        wav = _load_audio(wav_path, channel=channel)
        inputs = processor(wav, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)
            feat = output.last_hidden_state.squeeze(0)
            if output_norm:
                feat = F.layer_norm(feat, feat.shape[-1:])

        arr = feat.detach().cpu().numpy()
        if dtype == "float16":
            arr = arr.astype(np.float16)
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)


def main():
    parser = argparse.ArgumentParser(description="Extract frozen SSL frame-level features.")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")
    args = parser.parse_args()
    extract_features(load_config(args.config))


if __name__ == "__main__":
    main()

