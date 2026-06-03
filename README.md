# CSE 144 Final Project

[Kaggle competition](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project)

## Team

<!-- Fill in before submission -->
- Name:
- Name:

## What's in the repo

| Path | Description |
|------|-------------|
| `CSE-144_Final_Project.ipynb` | Training and inference (EfficientNet-B2, two-phase fine-tune, 5-view TTA) |
| `submission.csv` | Shared group predictions (1036 rows: 1000 sample IDs + 36 extra test images) |
| `data/sample_submission.csv` | Kaggle submission format reference |
| `checkpoints/best_model.pt` | Best weights (phase 2 epoch 9, val acc **75.47%**) — commit this file so teammates can skip training |
| `requirements.txt` | Python dependencies |

**Not in the repo (download locally):** `data/train/` and `data/test/` image folders from Kaggle.

## Setup

```bash
pip install -r requirements.txt
```

Download competition data from Kaggle and place it here:

```
data/
  train/          # class folders 0..99
  test/           # unlabeled .jpg files
  sample_submission.csv
```

## Quick start (Kaggle only)

If you only need to submit the shared predictions:

1. Upload `submission.csv` on the competition **Submit Predictions** page.

No training required.

## Full pipeline (notebook)

Open `CSE-144_Final_Project.ipynb` and run:

1. **Setup** — imports, paths, hyperparameters  
2. **Data** — loaders (requires `data/train/`)  
3. **Model** — EfficientNet-B2 + custom head  
4. **Training** — phase 1: frozen backbone (8 epochs); phase 2: full fine-tune with mixup (up to 45 epochs; best checkpoint saved automatically)  
5. **Submission** — loads `checkpoints/best_model.pt`, runs 5-view TTA, writes `submission.csv`

To regenerate `submission.csv` without retraining, run setup → data → model → **submission** (skip training). The submission cell loads `checkpoints/best_model.pt`.

**Note:** Training is slow on CPU; use GPU if available. Lower `BATCH_SIZE` in the notebook if you run out of memory.

## Model summary

- **Backbone:** ImageNet EfficientNet-B2  
- **Training:** RandAugment, RandomErasing, label smoothing, mixup (phase 2), AdamW with OneCycle (phase 1) and cosine warmup/decay (phase 2)  
- **Best validation accuracy:** 75.47% (5% hold-out split; noisy but used for checkpointing)  
- **Kaggle test accuracy:** scored only after upload (labels are hidden locally)

## Leaderboard

After submitting on Kaggle, add a screenshot at `docs/leaderboard.png` and link it here:

<!-- Optional: ![Leaderboard](docs/leaderboard.png) -->
