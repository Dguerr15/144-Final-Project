# CSE 144 Final Project

[Kaggle competition](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project)

## Team

- Name:
- Name:

## What's in the repo

| Path | Description |
|------|-------------|
| `CSE-144_Final_Project.ipynb` | Training and inference notebook |
| `submission.csv` | Group predictions for Kaggle (1036 rows) |
| `checkpoints/best_model_On_Kaggle.pt` | Best weights used for submission (epoch 21) |
| `data/sample_submission.csv` | Kaggle format reference |
| `requirements.txt` | Python dependencies |

**Download locally:** `data/train/` and `data/test/` from Kaggle (not committed).

## Setup

```bash
pip install -r requirements.txt
```

Place competition data here:

```
data/
  train/          # folders 0..99
  test/           # unlabeled .jpg files
  sample_submission.csv
```

## Quick submit

Upload `submission.csv` on the Kaggle **Submit Predictions** page. No training needed.

## Notebook workflow

Open `CSE-144_Final_Project.ipynb` and run:

1. **Setup** — imports, paths, hyperparameters
2. **Data** — loaders and train/val split (needs `data/train/`)
3. **Model** — EfficientNet-V2-S with a custom classifier head
4. **Training** — phase 1 warms up the head (6 epochs), phase 2 fine-tunes the full model (up to 50 epochs with early stopping)
5. **Plots** — loss and accuracy curves (optional)
6. **Submission** — loads `checkpoints/best_model_On_Kaggle.pt`, runs 5-view TTA, writes `submission.csv`

To regenerate `submission.csv` without retraining, run setup → data → model → submission.

Training is slow on CPU. Lower `BATCH_SIZE` if you run out of memory.

## Model summary

- **Backbone:** ImageNet EfficientNet-V2-S
- **Validation:** stratified split, 1 image per class (100 images)
- **Augmentation:** random resized crop, flip, color jitter, random erasing
- **Phase 2:** mixup/cutmix, label smoothing, AdamW, cosine LR schedule
- **Inference:** 5-view test-time augmentation
- **Best val accuracy:** 71% (epoch 21)
- **Kaggle test accuracy:** 76%
