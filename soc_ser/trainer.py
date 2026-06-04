import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import FeatureDataset, feature_collate
from .metrics import compute_metrics
from .model import SOCClassifier
from .utils import resolve_device, save_json, set_seed


def _make_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=feature_collate)


def _evaluate(model, loader, criterion, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    preds: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    metrics = compute_metrics(labels, preds, num_classes)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def _linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _num_classes(label_map: Dict[str, int]) -> int:
    labels = sorted(set(label_map.values()))
    expected = list(range(len(labels)))
    if labels != expected:
        raise ValueError(f"label_map values must be contiguous class ids {expected}, got {labels}")
    return len(labels)


def train_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(config.get("seed", 2024))
    set_seed(seed)

    device = resolve_device(config.get("device", "auto"))
    label_map = config["label_map"]
    num_classes = _num_classes(label_map)
    base_dir = config["base_dir"]
    feature_dir = config["feature_dir"]
    file_prefix = config["file_prefix"]
    num_folds = int(config.get("num_folds", 5))
    output_dir = config.get("output_dir", "outputs/soc")
    os.makedirs(output_dir, exist_ok=True)

    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 100))
    patience = int(config.get("patience", 15))
    val_ratio = float(config.get("val_ratio", 0.2))
    lr = float(config.get("lr", 5e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    grad_clip = config.get("grad_clip", 1.0)

    fold_results = []
    for fold_idx in range(1, num_folds + 1):
        fold_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        train_jsonl = os.path.join(fold_dir, f"{file_prefix}_train_fold_{fold_idx}.jsonl")
        test_jsonl = os.path.join(fold_dir, f"{file_prefix}_test_fold_{fold_idx}.jsonl")

        train_full = FeatureDataset(train_jsonl, feature_dir, label_map)
        test_set = FeatureDataset(test_jsonl, feature_dir, label_map)

        val_len = max(1, int(len(train_full) * val_ratio))
        train_len = len(train_full) - val_len
        if train_len < 1:
            raise RuntimeError("Training split is empty. Increase training data or reduce val_ratio.")

        train_set, val_set = random_split(
            train_full,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = _make_loader(train_set, batch_size, shuffle=True)
        val_loader = _make_loader(val_set, batch_size, shuffle=False)
        test_loader = _make_loader(test_set, batch_size, shuffle=False)

        model = SOCClassifier(
            input_dim=int(config.get("input_dim", 768)),
            spd_dim=int(config.get("spd_dim", 24)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_classes=num_classes,
            dropout=float(config.get("dropout", 0.4)),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = max(len(train_loader) * epochs, 1)
        scheduler = _linear_warmup_decay(
            optimizer,
            warmup_steps=int(total_steps * warmup_ratio),
            total_steps=total_steps,
        )

        best_val = float("inf")
        best_path = os.path.join(output_dir, f"fold_{fold_idx}_best.pth")
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}", leave=False)
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
                scheduler.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_metrics = _evaluate(model, val_loader, criterion, device, num_classes)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(model.state_dict(), best_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        model.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = _evaluate(model, test_loader, criterion, device, num_classes)
        result = {"fold": fold_idx, **{k: test_metrics[k] for k in ["WA", "UA", "F1"]}}
        fold_results.append(result)
        print(
            f"Fold {fold_idx}: "
            f"WA={result['WA']:.4f}, UA={result['UA']:.4f}, F1={result['F1']:.4f}"
        )

    summary = {
        "folds": fold_results,
        "average": {
            "WA": sum(x["WA"] for x in fold_results) / len(fold_results),
            "UA": sum(x["UA"] for x in fold_results) / len(fold_results),
            "F1": sum(x["F1"] for x in fold_results) / len(fold_results),
        },
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))
    return summary
