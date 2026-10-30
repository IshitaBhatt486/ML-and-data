"""Backward-compatible entry point for training the bovine-breed classifier.

The original notebook export depended on a personal Colab path and was not a
valid standalone Python program. Use this script exactly as the documented
``python -m src.train`` command, for example::

    python cifar100_resnet.py --data-dir data/Indian_bovine_breeds --epochs 50
"""

from src.train import main


if __name__ == "__main__":
    main()
