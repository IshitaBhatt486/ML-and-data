# Indian Bovine Breed Classification

Classify Indian cattle and buffalo breeds from images with a custom TensorFlow ResNet-50 model.

This project was initially made as a part of a competition. The reusable training code now lives in `src/`; the original notebook is kept for reference.

## What is included

- A custom ResNet-50 implementation for multi-class image classification
- Reproducible train/validation split with TensorFlow data pipelines
- Image augmentation, checkpointing, early stopping, TensorBoard logs, and training-history export
- Evaluation script that reports loss and accuracy
- The original Colab notebook: `cifar100_resnet.ipynb`

## Project layout

```text
.
├── cifar100_resnet.ipynb    # Exploratory notebook
├── cifar100_resnet.py       # Notebook export
├── requirements.txt
├── src/
│   ├── data.py              # Dataset loading and augmentation
│   ├── model.py             # Custom ResNet-50 architecture
│   ├── train.py             # Training entry point
│   └── evaluate.py          # Evaluation entry point
└── README.md
```

## Dataset

Use a folder-per-class image dataset. The expected layout is:

```text
data/Indian_bovine_breeds/
├── Gir/
│   ├── image_001.jpg
│   └── ...
├── Murrah/
│   └── ...
└── Sahiwal/
    └── ...
```

## Setup

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Train

Place the dataset in `data/Indian_bovine_breeds/`, then run:

```bash
python -m src.train --data-dir data/Indian_bovine_breeds --epochs 50
```

```bash
python -m src.train --help
python -m src.train --data-dir data/Indian_bovine_breeds --image-size 128 --batch-size 16
```

Training creates `models/best_model.keras`, `models/final_model.keras`, `models/class_names.json`, `models/history.json`, and TensorBoard logs in `logs/`.

```bash
tensorboard --logdir logs
```

## Evaluate

```bash
python -m src.evaluate --data-dir data/Indian_bovine_breeds --model models/best_model.keras
```

The validation split is deterministic (`seed=42`), so evaluation uses the same held-out images as training.

# Credits
- Kaggle dataset: [Indian Bovine Breeds dataset on Kaggle](https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds).

Team Members:
- Sai Prashanth
- Hasveel Singh
- Ishita Bhatt
- Paranjai Pareshad
- Ryan Chatri
- Harshit Sharma