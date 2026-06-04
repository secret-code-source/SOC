# Data

This directory is intentionally empty.

Prepare EmoBox-style metadata and extracted SSL features before training. The
default configs expect:

```text
data/
  ravdess/
    ravdess.jsonl
    fold_1/ravdess_train_fold_1.jsonl
    fold_1/ravdess_test_fold_1.jsonl
    ...
  esd/
    esd.jsonl
    fold_1/esd_train_fold_1.jsonl
    fold_1/esd_test_fold_1.jsonl
    ...
  features/
    ravdess_hubert/*.npy
    esd_hubert/*.npy
```

