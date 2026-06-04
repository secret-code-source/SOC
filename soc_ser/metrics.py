from typing import Dict, List

import numpy as np


def compute_metrics(labels: List[int], preds: List[int], num_classes: int) -> Dict[str, float]:
    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    wa = float((labels_arr == preds_arr).mean()) if len(labels_arr) else 0.0

    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = int(((preds_arr == cls) & (labels_arr == cls)).sum())
        fp = int(((preds_arr == cls) & (labels_arr != cls)).sum())
        fn = int(((preds_arr != cls) & (labels_arr == cls)).sum())
        support = tp + fn
        if support > 0:
            recalls.append(tp / support)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / support
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)

    return {
        "WA": wa,
        "UA": float(np.mean(recalls)) if recalls else 0.0,
        "F1": float(np.mean(f1s)) if f1s else 0.0,
    }

