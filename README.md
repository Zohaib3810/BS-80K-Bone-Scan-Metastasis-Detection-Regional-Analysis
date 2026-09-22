
# BS-80K: Bone Scan Metastasis Detection & Regional Analysis

This repository implements an end-to-end deep learning pipeline for detecting bone metastases in Whole Body scans and regional bone scans from the **BS-80K dataset**. It utilizes a hybrid architecture featuring a PyTorch-based **Swin Transformer** backbone for robust feature extraction and a **TensorFlow/Keras** classification/detection head, optimized to handle high class imbalances.

## Project Overview
Metastasis detection in bone scintigraphy (bone scans) is critical for cancer staging. This project presents a full workflow to handle medical imaging challenges including:
- **Preprocessing**: Median Filtering for denoising and Contrast Limited Adaptive Histogram Equalization (CLAHE) for localized contrast enhancement.
- **Spatial Slicing & Padding**: Resizing preserving anatomical aspect ratios, and regional slicing of whole-body scans into functional anatomical zones.
- **Multi-View Fusion**: Combining Anterior (ANT) and Posterior (POST) views using Early Fusion (channel stacking) and Late Fusion (multi-stream model architecture).
- **Class Imbalance Mitigation**: Implemented using patient-level stratified splitting (`StratifiedGroupKFold`), Weighted BCE Loss, and Focal Loss.
- **Explainable AI (XAI)**: Highlighting pathological regions of interest using Swin Self-Attention Maps.

---

## Directory & Dataset Structure
The project expects the dataset to be downloaded via `kagglehub` from [mariusmarin/bs-80k](https://www.kaggle.com/datasets/mariusmarin/bs-80k) with the following anatomy:

```
/kaggle/input/bs-80k/temp/
│
├── ankleRPOST/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ankleRPOST.txt  <-- Maps filenames to binary labels (0: Normal, 1: Abnormal)
├── wholeBodyANT/
└── ... (28 subdirectories matching anatomical views)
```

---

## Pipeline Steps

### 1. Data Loading & Stratified Splitting
- Scans are dynamically paired and enriched with folder metadata.
- To prevent **Data Leakage**, images are split using `StratifiedGroupKFold` grouped by patient/scan base filename. This ensures that a patient's ANT and POST views do not cross-contaminate training and validation splits.

### 2. Image Preprocessing
- **Denoising**: Grayscale images are smoothed using a 3x3 Median Filter.
- **Contrast Enhancement**: CLAHE is applied (clip limit: `2.0`, grid size: `8x8`) to highlight small hot-spots of abnormal tracer accumulation.
- **Multi-View Fusion**: Stacks ANT and POST views as a 2-channel tensor inputs for dual-aspect diagnostic logic.

### 3. Feature Extraction
- Leverages a pre-trained **Swin Transformer** (`swin_tiny_patch4_window7_224` via `timm`) as a powerful frozen backbone.
- To optimize computational cost, features are **pre-extracted** and stored as compressed `.npz` files on disk, accelerating down-stream training epochs by over **10x**.

### 4. Classification & Object Detection Heads
- **Binary Classifier**: Features a classification head with Dense layers, Dropout (`0.3`), and a single sigmoid output trained with Weighted BCE to penalize positive class misses.
- **Detection Head (RetinaNet style)**: Formulates bounding box regression and anchor-based classification, optimized with **Focal Loss** to counteract dense negative backgrounds.

### 5. Interpretability (XAI)
- The last layer of the Swin Transformer is probed to extract global attention maps.
- These attention maps are superimposed over the original scans as heatmaps, making the model's diagnostic decisions explainable and clinical-grade.

---

## Performance & Training Metrics

The classifier is trained with a dynamic threshold search at validation time to optimize the **F1-Score** directly:
- **Train Loss / Val Loss**: Decreasing stably over 15 epochs.
- **Train AUC**: ~0.95+
- **Validation F1**: Evaluated dynamically using threshold tuning (optimized to prioritize cancer sensitivity).

*The loss and metric visualizations are automatically generated and displayed in the notebook after training concludes.*

---

## How to Run

1. **Environment Setup**:
   Install dependencies:
   ```bash
   pip install kagglehub pandas opencv-python tensorflow torch torchvision timm scikit-learn matplotlib
   ```

2. **Run the Notebook**:
   Open the Colab/Jupyter notebook and execute cells sequentially. The dataset will auto-download using `kagglehub` into your environment's scratch space.

3. **View Preextracted Features**:
   Extracted Swin-features will be saved under the directory `preextracted_features/` as compressed NumPy arrays.

---

## License
This project is distributed under the MIT License.
```
