# Seizure Status Predictor

An end-to-end Machine Learning and Deep Learning pipeline that classifies short
windows of scalp-EEG as **seizure** or **normal** from raw `.edf` recordings. The
project takes continuous multi-channel EEG, slices it into 4-second windows, labels
each window using the dataset's seizure annotations, and trains two different models
so their performance can be compared:

1. **Random Forest baseline** on hand-crafted frequency-band features (Welch's PSD / Fourier transform).
2. **1D Convolutional Neural Network (PyTorch)** that learns directly from the raw waveform.

---

## 📊 The Data Challenge: Severe Class Imbalance

The recordings come from the [Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/)
(patient `PN00`). Each recording is a multi-channel EEG sampled at **512 Hz**, paired
with a text file listing the start/end times of each seizure. A primary challenge of
this medical dataset is severe class imbalance, where normal brain activity drastically
outnumbers seizure events.

| Property         | Value                                              |
| ---------------- | -------------------------------------------------- |
| Channels         | 35                                                 |
| Window length    | 4 s → **2048 samples** per window                  |
| Sampling rate    | 512 Hz                                             |
| Band-pass filter | 0.5 – 60 Hz                                         |
| Class balance    | Heavily imbalanced (~2.7% of windows are seizures) |

Per-split class counts:

* **Validation Set:** 506 Normal chunks vs. 19 Seizure chunks
* **Test Set:** 518 Normal chunks vs. 17 Seizure chunks

Recording `PN00-3` is skipped because it is corrupted. The remaining files are split by
recording to avoid leakage:

| Split      | Recording(s)       |
| ---------- | ------------------ |
| Train      | `PN00-1`, `PN00-2` |
| Validation | `PN00-4`           |
| Test       | `PN00-5`           |

> **Note:** The raw `.edf` data and the generated `*.npy` arrays are git-ignored, so
> they are not included in this repo. Place the Siena `PN00` files under
> `Seizure-EEG-Data/` to regenerate them (see below).

---

## 🚀 Model Architectures & Methodologies

### 1. Random Forest + Fourier Features (Scikit-Learn)
* **Feature Engineering:** Raw EEG signals are processed using Welch's method to extract Power Spectral Density (PSD) across five clinical brainwave frequency bands: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–12 Hz), Beta (12–30 Hz), and Gamma (30–60 Hz).
* **Classification:** A 100-tree Random Forest Classifier trains on these flat frequency-domain arrays.

### 2. 1D Convolutional Neural Network (PyTorch)
* **Architecture:** Slides filters directly over the raw time-steps across all 35 channels simultaneously, detecting temporal spike shapes without crushing the timeline sequence.
* **Regularization:** Employs `BatchNorm1d` and `Dropout` (30%) layers to curb overfitting.
* **Imbalance Handling:** Uses `BCEWithLogitsLoss` with a `pos_weight=25.0` penalty scalar to heavily punish missed seizures during training.

---

## 🏆 Head-to-Head Performance Results

Evaluated strictly on the unseen **Final Exam Test Set (`PN00-5`)**, we focus heavily on
**Label 1 (Seizure)** metrics since overall accuracy is skewed by the class imbalance.

| Metric (Seizure Class) | Random Forest (Fourier Features) | 1D-CNN (Raw Timeline) |
| :--- | :---: | :---: |
| **Precision** | **1.00** | **0.69** |
| **Recall (Sensitivity)** | **0.71** | **0.53** |
| **F1-Score** | **0.83** | **0.60** |
| **Overall Accuracy** | **99.07%** | **97.76%** |

### Key Findings & Takeaways
1. **Domain Knowledge Wins:** The Random Forest model achieved a superior Recall of **71%** (catching 12 out of 17 actual seizures) with perfect precision (zero false alarms). This underscores the massive value of using traditional signal-processing features (Welch's PSD) for smaller medical datasets.
2. **Deep Learning Data Hunger:** The PyTorch 1D-CNN struggled to generalize as effectively, achieving a **53%** recall. While the training loss dropped smoothly to `0.0060`, the limited number of positive seizure samples made it prone to local overfitting rather than mastering universal temporal wave sequences.

---

## 📈 Visualizations Included

Two optional exploratory plots (`plot_frequency_spectrum`, `plot_feature_bars`) help
visualize how the frequency content differs between normal and seizure windows:
* **Frequency Spectrums:** Side-by-side spectral comparisons demonstrating how the Fourier transform isolates power spikes during active seizures compared to normal waves.
* **Feature Importance Bars:** Visualizes the jump in average power across the delta and theta ranges during seizure activity.
* **Confusion Matrices:** Multi-panel plots illustrating exactly where false positives and false negatives hit the validation and test sets.

The plots open interactive windows, so they are left commented out at the bottom of
`modeling.py` — uncomment them to use.

---

## Pipeline

### 1. Data preparation — `data_handling.ipynb`

Loads the `.edf` recordings with [MNE](https://mne.tools/), band-pass filters them,
parses the seizure annotation file into per-recording timelines, then chops each
recording into non-overlapping 4-second windows and labels every window
(`1` if it overlaps a seizure, otherwise `0`). The resulting arrays are saved as
`X_train.npy`, `Y_train.npy`, `X_val.npy`, `Y_val.npy`, `X_test.npy`, `Y_test.npy`.

### 2. Modeling & comparison — `modeling.py`

Loads the prepared arrays and runs both models:

- **`skl_pipeline`** — extracts mean power in five frequency bands
  (Delta, Theta, Alpha, Beta, Gamma) per channel using Welch's method, then trains
  a `RandomForestClassifier` and reports accuracy + a classification report on the
  validation and test sets, with confusion matrices.
- **`neural_network_pipeline`** — z-score normalizes the raw windows and trains
  `SeizureCNN`, a 1D CNN with batch norm and dropout. Because seizures are rare, the
  loss uses `BCEWithLogitsLoss(pos_weight=25)` to penalize missed seizures more heavily.

---

## 🛠️ Usage

```bash
# 1. Install dependencies
pip install mne numpy pandas matplotlib scipy scikit-learn torch

# 2. Generate the train/val/test arrays
#    (open data_handling.ipynb and run all cells)

# 3. Train and compare both models
python modeling.py
```

## Project structure

```
.
├── data_handling.ipynb   # EEG loading, windowing, labeling → .npy splits
├── modeling.py           # Random Forest baseline + PyTorch CNN comparison
├── Seizure-EEG-Data/     # raw .edf recordings + seizure annotations (git-ignored)
└── *.npy                 # generated train/val/test arrays (git-ignored)
```
