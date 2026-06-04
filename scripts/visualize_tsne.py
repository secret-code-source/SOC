import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from soc_ser.dataset import FeatureDataset, feature_collate
from soc_ser.model import SOCClassifier
from soc_ser.utils import load_config, resolve_device


def _plot(embeddings, labels, id_to_label, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 6), dpi=180)
    for cls_id, name in id_to_label.items():
        mask = labels == cls_id
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], s=18, alpha=0.75, label=name)
    plt.xticks([])
    plt.yticks([])
    plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved t-SNE figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SOC embeddings with t-SNE.")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="figures/soc_tsne.png")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.get("device", "auto"))
    label_map = config["label_map"]
    id_to_label = {v: k for k, v in label_map.items()}

    fold_dir = os.path.join(config["base_dir"], f"fold_{args.fold}")
    test_jsonl = os.path.join(fold_dir, f"{config['file_prefix']}_test_fold_{args.fold}.jsonl")
    dataset = FeatureDataset(test_jsonl, config["feature_dir"], label_map)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=feature_collate)

    model = SOCClassifier(
        input_dim=int(config.get("input_dim", 768)),
        spd_dim=int(config.get("spd_dim", 24)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_classes=len(id_to_label),
        dropout=float(config.get("dropout", 0.4)),
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    feats = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting SOC embeddings"):
            z = model.pool(x.to(device))
            feats.append(z.cpu().numpy())
            labels.extend(y.numpy().tolist())

    feats = np.concatenate(feats, axis=0)
    labels = np.asarray(labels)
    feats = StandardScaler().fit_transform(feats)
    if feats.shape[1] > 50:
        feats = PCA(n_components=50, random_state=42).fit_transform(feats)
    perplexity = min(args.perplexity, max(2.0, (len(feats) - 1) / 3))
    emb = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=42,
    ).fit_transform(feats)

    _plot(emb, labels, id_to_label, args.output)


if __name__ == "__main__":
    main()

