# Seizure Status Predictor

An end-to-end Machine Learning and Deep Learning pipeline designed to classify seizure activity from multi-channel EEG time-series data. This project implements a classical machine learning baseline using handcrafted frequency domain features alongside a raw-timeline 1D Convolutional Neural Network (1D-CNN) built in PyTorch.

## 📊 The Data Challenge: Severe Class Imbalance
The dataset consists of 35-channel EEG recordings sliced into chunks of 2,048 time steps (sampled at 512 Hz). A primary challenge of this medical dataset is severe class imbalance, where normal brain activity drastically outnumbers seizure events:

* **Validation Set:** 506 Normal chunks vs. 19 Seizure chunks
* **Test Set:** 518 Normal chunks vs. 17 Seizure chunks

---

## 🚀 Model Architectures & Methodologies

### 1. Random Forest + Fourier Features (Scikit-Learn)
* **Feature Engineering:** Raw EEG signals are processed using Welch's method to extract Power Spectral Density (PSD) across five clinical brainwave frequency bands: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–12 Hz), Beta (12–30 Hz), and Gamma (30–60 Hz).
* **Classification:** A 100-tree Random Forest Classifier trains on these flat frequency-domain arrays.

### 2. 1D Convolutional Neural Network (PyTorch)
* **Architecture:** Slid filters directly over the raw time-steps across all 35 channels simultaneously, detecting temporal spike shapes without crushing timeline sequence.
* **Regularization:** Employs `BatchNorm1d` and `Dropout` (30%) layers to curb overfitting.
* **Imbalance Handling:** Utilizes `BCEWithLogitsLoss` equipped with a `pos_weight=25.0` penalty scalar to heavily punish missed seizures during training.

---

## 🏆 Head-to-Head Performance Results

Evaluated strictly on the unseen **Final Exam Test Set (File 5)**, we focus heavily on **Label 1 (Seizure)** metrics since overall accuracy is skewed by the class imbalance.

| Metric (Seizure Class) | Random Forest (Fourier Features) | 1D-CNN (Raw Timeline) |
| :--- | :---: | :---: |
| **Precision** | **1.00** | **0.69** |
| **Recall (Sensitivity)** | **0.71** | **0.53** |
| **F1-Score** | **0.83** | **0.60** |
| **Overall Accuracy** | **99.07%** | **97.76%** |

### Key Findings & Key Takeaways
1. **Domain Knowledge Wins:** The Random Forest model achieved a superior Recall of **71%** (catching 12 out of 17 actual seizures) with perfect precision (zero false alarms). This underscores the massive value of using traditional signal processing features (Welch's PSD) for smaller medical datasets.
2. **Deep Learning Data Hunger:** The PyTorch 1D-CNN struggled to generalize as effectively, achieving a **53%** recall. While the training loss dropped smoothly to `0.0060`, the limited number of positive seizure samples made it prone to local overfitting rather than mastering universal temporal wave sequences.

---

## 📈 Visualizations Included

The script dynamically generates multiple diagnostic plots during execution:
* **Frequency Spectrums:** Side-by-side spectral comparisons demonstrating how the Fourier Transform isolates power spikes during active seizures compared to normal waves.
* **Feature Importance Bars:** Visualizes the massive jump in average loudness/power across delta and theta ranges during seizure activity.
* **Confusion Matrices:** Multi-panel plots illustrating exactly where false positives and false negatives hit the validation and testing sets.

---

## 🛠️ How to Run the Pipeline

1. Clone the repository and ensure your local hard drive contains your data arrays (`X_train.npy`, `Y_train.npy`, etc.).
2. Install dependencies:
   ```bash
   pip install torch numpy pandas scikit-learn scipy matplotlib
