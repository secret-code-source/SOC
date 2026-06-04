# t-SNE Visualization

This optional utility visualizes SOC embeddings from a trained checkpoint.

Install visualization-only dependencies:

```bash
pip install -r requirements-viz.txt
```

Run after training:

```bash
python scripts/visualize_tsne.py \
  --config configs/ravdess_hubert.yaml \
  --fold 1 \
  --checkpoint outputs/ravdess_hubert_soc/fold_1_best.pth \
  --output figures/ravdess_fold1_tsne.png
```

The script loads the test split for the selected fold, extracts SOC embeddings
with `model.pool`, applies PCA when needed, and saves a 2D t-SNE plot.

This visualization is qualitative and is not required for training or evaluation.

