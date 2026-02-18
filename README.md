# MKMGCN-DDI: Multi-Kernel Magnetic Graph Convolutional Network for Drug-Drug Interaction Prediction

## Overview

MKMGCN-DDI is a deep learning method for predicting drug-drug interactions (DDIs). The model employs a Multi-Kernel Magnetic Graph Convolutional Network to capture complex relationships between drugs and comprehensively predict their interactions.

## Quick Start

The main entry point for training and evaluating the model is **`main.py`**.

To run the model:
```bash
python main.py
```

## Running Instructions

**Main script**: `main.py` contains all the training and evaluation logic.

Basic usage:
```bash
python main.py --task type --epochs 500 --lr 0.01
```

Available command-line arguments:
- `--model`: Model name (default: 'MKMGCN-DDI')
- `--epochs`: Number of training epochs (default: 500)
- `--lr`: Learning rate (default: 0.01)
- `--weight_decay`: Weight decay (default: 0)
- `--task`: Task type - 'type', 'direction', 'direction-specific', 'joint-4C', 'joint-5C' (default: 'type')
- `--dropout`: Dropout rate (default: 0.5)
- `--hidden`: Hidden dimension (default: 32)
- `--train_ratio`: Training set ratio (default: 0.8)
- `--test_ratio`: Test set ratio (default: 0.2)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chen2025mkmgcn,
  title={MKMGCN-DDI: Predicting drug-drug interactions via magnetic graph convolutional network with multiple kernels},
  author={Chen, Ming and Pan, Yunhan and Lei, Xiujuan and Ji, Chunyan and Dai, Yinglong and Pan, Yi},
  journal={IEEE Transactions on Computational Biology and Bioinformatics},
  volume={22},
  number={2},
  pages={469--480},
  year={2025},
  publisher={IEEE}
}
```

