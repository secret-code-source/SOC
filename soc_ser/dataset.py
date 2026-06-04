import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, jsonl_path: str, feature_dir: str, label_map: Dict[str, int]):
        self.items: List[Tuple[str, int]] = []
        self.jsonl_path = jsonl_path
        self.feature_dir = feature_dir
        self.label_map = label_map

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Metadata file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = item.get("key") or item.get("sid") or item.get("id")
                label = item.get("emo") or item.get("emotion") or item.get("label")
                if key is None or label not in label_map:
                    continue
                feature_path = os.path.join(feature_dir, f"{key}.npy")
                if os.path.exists(feature_path):
                    self.items.append((feature_path, label_map[label]))

        if not self.items:
            raise RuntimeError(
                f"No usable samples found for {jsonl_path}. "
                f"Check feature_dir={feature_dir} and label_map keys."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        feature_path, label = self.items[index]
        feat = np.load(feature_path)
        if feat.ndim == 3:
            feat = feat[-1]
        if feat.ndim != 2:
            raise ValueError(f"Expected [T, D] feature, got shape {feat.shape}: {feature_path}")
        return torch.from_numpy(feat).float(), label


def feature_collate(batch):
    feats, labels = zip(*batch)
    return pad_sequence(feats, batch_first=True), torch.tensor(labels, dtype=torch.long)

