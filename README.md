# 🛰️ Rural Image Segmentation On-Board

> **Semantic segmentation for autonomous navigation in rural environments** — optimized for on-board, resource-constrained inference.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com)

---

## 📖 Overview

This project implements a pixel-wise **semantic segmentation** pipeline for images captured by a forward-facing camera mounted on a vehicle navigating rural, off-road environments. The goal is to classify every pixel into one of **9 semantic classes** representing terrain types and obstacles — enabling downstream path planning and autonomous navigation decisions.

The system is built around a **ResNet18 encoder + U-Net decoder** architecture, chosen for its strong accuracy-to-efficiency trade-off. The design satisfies strict on-board deployment constraints: **< 4 GB GPU RAM** at inference time and **< 5 GB GPU RAM** during training, while maintaining competitive per-class IoU.

University of Salerno — Machine Learning course project, A.Y. 2024/25.

---

## 🗺️ Semantic Classes

The model classifies each pixel into one of the following 9 classes:

| ID | Class | Color | RGB |
|----|-------|-------|-----|
| 0 | Background | ⬜ White | `(255, 255, 255)` |
| 1 | Sky | 🟦 Blue | `(1, 88, 255)` |
| 2 | Rough path | 🟫 Brown | `(156, 76, 30)` |
| 3 | Smooth path | ◼ Gray | `(178, 176, 153)` |
| 4 | Walkable grass | 🟩 Light green | `(128, 255, 0)` |
| 5 | Tall vegetation | 🌲 Dark green | `(40, 80, 0)` |
| 6 | Low non-walkable vegetation | 🟢 Medium green | `(0, 160, 0)` |
| 7 | Puddle | 🟣 Fuchsia | `(255, 0, 128)` |
| 8 | Obstacle | 🔴 Red | `(255, 0, 0)` |

---

## 📊 Qualitative Results

*Visualization grid — Original image · Ground Truth · Prediction*

<!-- 
  ════════════════════════════════════════════════════════════════
  ADD YOUR RESULTS IMAGE HERE
  ────────────────────────────────────────────────────────────────
  1. Generate your grid (Original | GT | Prediction) using the
     visualization cell in the notebook.
  2. Save the image (e.g., results_grid.png).
  3. Create an "assets/" folder in this repository and upload it.
  4. Replace the placeholder below with the actual path:
     ![Results Grid](assets/results_grid.png)
  ════════════════════════════════════════════════════════════════
-->

> 📌 *Results visualization coming soon — add your grid image to `assets/results_grid.png`*

---

## 🏗️ Architecture

The segmentation model follows a **fully convolutional encoder-decoder** design with skip connections:

```
Input (H × W × 3)
       │
┌──────▼──────┐
│  ResNet18   │  ← ImageNet pretrained encoder
│  Encoder   │     Extracts hierarchical spatial features
│ (5 stages) │     Stride: ×2, ×4, ×8, ×16, ×32
└──────┬──────┘
       │  bottleneck features
┌──────▼──────┐
│   U-Net     │  ← Decoder with skip connections
│   Decoder  │     Progressive upsampling back to H × W
│ (4 stages) │     Skip connections preserve fine-grained detail
└──────┬──────┘
       │
Output (H × W × 9)   ← One probability map per class
       │
    argmax → Predicted label map (H × W × 1, uint8)
```

**Why ResNet18 + U-Net?**
- ResNet18 provides a strong pretrained feature extractor at minimal parameter cost (~11M params)
- U-Net skip connections recover spatial resolution lost during downsampling — critical for precise boundary segmentation
- The fully convolutional design satisfies the < 4 GB GPU RAM inference constraint
- Higher-capacity encoders (ResNet34/50, EfficientNet) were evaluated but exceeded the memory budget

---

## 📁 Dataset

| Property | Details |
|----------|---------|
| Source | University of Salerno (private dataset) |
| Total samples | 931 (image + label map pairs) |
| Image acquisition | Single fixed camera, rural outdoor environment |
| Conditions | Variable illumination (daylight only) |
| Label format | Grayscale mask, uint8 (values 0–8) |
| Split strategy | Train / Validation (custom split, no test labels released) |
| Augmentation | Online augmentation during training (flips, color jitter, random crops) |
| Constraint | Dataset extension **not** allowed |

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Backbone | ResNet18 (ImageNet pretrained) |
| Loss function | Weighted Cross-Entropy (class frequency weighting) |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Batch size | 8 |
| Epochs | *(see notebook)* |
| Input resolution | *(see notebook)* |

**Class imbalance handling:** The dataset has significant class imbalance (background and sky dominate). A per-class frequency-based loss weighting was applied to prevent the model from ignoring rare but critical classes (puddles, obstacles).

---

## 📈 Evaluation Metric

Performance is measured via **mean per-class Intersection-over-Union (mIoU)**:

$$\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$

$$\text{mIoU} = \frac{1}{C} \sum_{c=0}^{C-1} \text{IoU}_c$$

The final score balances three factors:
- **mIoU** — segmentation accuracy (higher is better)
- **GPU RAM at inference** — memory footprint (< 4 GB required)
- **Frame rate** — throughput on a fixed GPU (higher is better)

---

## 🚀 Getting Started

### Requirements

```bash
pip install torch torchvision numpy matplotlib Pillow
```

The notebook is designed to run on **Google Colab** (GPU runtime recommended).

### Run the Notebook

1. Open `Progetto_25_06_ResNet18_UNet_FileSalvati.ipynb` in Google Colab
2. Mount Google Drive and set the dataset path
3. Run all cells sequentially (training → evaluation → visualization)

### Inference (`predict` function)

The notebook exposes a `predict(X)` function compliant with the evaluation protocol:

```python
# X: np.ndarray of shape (batch_size, H, W, 3), dtype uint8
# returns: np.ndarray of shape (batch_size, H, W, 1), dtype uint8

predictions = predict(X)
```

The function handles all preprocessing (normalization, tensor conversion) and postprocessing (argmax, label remapping) internally, independently of batch size.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&color=525252)

---

## 📋 Repository Structure

```
├── Progetto_25_06_ResNet18_UNet_FileSalvati.ipynb   # Main training & evaluation notebook
├── test.ipynb                                        # Test script (predict function)
├── assets/                                           # ← Add results images here
│   └── results_grid.png                             # Visualization grid (to be added)
└── traccia.txt                                       # Original project specification
```

---

## 👤 Author

**Antonio Apicella** · [GitHub](https://github.com/apiantonio) · [LinkedIn](https://www.linkedin.com/in/antonio~apicella/)

*University of Salerno — Dept. of Information Engineering, Electrical Engineering and Applied Mathematics (DIEM)*
