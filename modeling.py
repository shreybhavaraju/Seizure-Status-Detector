import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from scipy.signal import welch

freq_ranges = {
    "Delta": [0.5,4.0],
    "Theta": [4.01,8.0],
    "Alpha": [8.01,12.0],
    "Beta": [12.01,30.0],
    "Gamma": [30.01,60.0]
}

def data_loading():
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")

    X_val = np.load("X_val.npy")
    Y_val = np.load("Y_val.npy")

    X_test = np.load("X_test.npy")
    Y_test = np.load("Y_test.npy")

    print("Data loaded perfectly inside modeling.py!")
    print(f"Train shapes: {X_train.shape}, {Y_train.shape}")
    print(f"Val shapes:   {X_val.shape}, {Y_val.shape}")
    print(f"Test shapes:  {X_test.shape}, {Y_test.shape}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def neural_network_pipeline( X_tr, Y_tr, X_val, Y_val, X_test, Y_test):
    pass

def skl_pipeline(X_tr, Y_tr, X_v, Y_v, X_t, Y_t):
    print("\n--- Starting Scikit-Learn + Fourier Pipeline ---")
    fs = 512
    
    # --- STEP 1: FEATURE EXTRACTION LOOP ---
    def extract_features(X_data):
        num_chunks = X_data.shape[0]
        num_channels = X_data.shape[1]
        X_flat = np.zeros((num_chunks, num_channels * 5))
        
        print(f"Extracting frequencies from {num_chunks} chunks...")
        for i in range(num_chunks):
            chunk_features = []
            for ch in range(num_channels):
                raw_wave = X_data[i, ch, :]
                frequencies, psd = welch(raw_wave, fs=fs, nperseg=512)
                
                delta = np.mean(psd[(frequencies >= 0.5) & (frequencies <= 4.0)])
                theta = np.mean(psd[(frequencies > 4.0) & (frequencies <= 8.0)])
                alpha = np.mean(psd[(frequencies > 8.0) & (frequencies <= 12.0)])
                beta  = np.mean(psd[(frequencies > 12.0) & (frequencies <= 30.0)])
                gamma = np.mean(psd[(frequencies > 30.0) & (frequencies <= 60.0)])
                
                chunk_features.extend([delta, theta, alpha, beta, gamma])
            X_flat[i] = chunk_features
        return X_flat

    X_train_flat = extract_features(X_tr)
    X_val_flat = extract_features(X_v)
    X_test_flat = extract_features(X_t)

    # --- STEP 2: TRAINING THE AI ---
    print("\nTraining the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_flat, Y_tr)

    # --- STEP 3: VALIDATION SET RESULTS ---
    val_preds = model.predict(X_val_flat)
    print(f"\n--- VALIDATION RESULTS (Practice Exam: File 4) ---")
    print(f"Accuracy: {accuracy_score(Y_v, val_preds) * 100:.2f}%")
    print(classification_report(Y_v, val_preds))
    
    # --- STEP 4: TEST SET RESULTS ---
    test_preds = model.predict(X_test_flat)
    print(f"\n--- TEST RESULTS (Final Exam: File 5) ---")
    print(f"Accuracy: {accuracy_score(Y_t, test_preds) * 100:.2f}%")
    print(classification_report(Y_t, test_preds))

    # --- GRAPHING THE RESULTS ---
    print("\nGenerating confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Validation Matrix
    cm_val = confusion_matrix(Y_v, val_preds)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["Normal", "Seizure"])
    disp_val.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title("Validation Set (File 4)")
    
    # Test Matrix
    cm_test = confusion_matrix(Y_t, test_preds)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Normal", "Seizure"])
    disp_test.plot(ax=axes[1], cmap=plt.cm.Greens)
    axes[1].set_title("Test Set (File 5)")
    
    plt.tight_layout()
    plt.show()

def plot_frequency_spectrum(X_tr, Y_tr):
    print("\n--- Generating Frequency Spectrum Graphs ---")
    fs = 512
    
    normal_idx = np.where(Y_tr == 0)[0][0]
    seizure_idx = np.where(Y_tr == 1)[0][0]
    
    # Grab Channel 0 for both chunks
    normal_wave = X_tr[normal_idx, 0, :]
    seizure_wave = X_tr[seizure_idx, 0, :]
    
    freqs_norm, psd_norm = welch(normal_wave, fs=fs, nperseg=512)
    freqs_seiz, psd_seiz = welch(seizure_wave, fs=fs, nperseg=512)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Graph 1: Normal Brainwave Frequencies
    axes[0].plot(freqs_norm, psd_norm, color='green', label='Normal (Label 0)')
    axes[0].set_title("Normal Chunk Frequency Spectrum")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Loudness / Power (PSD)")
    axes[0].set_xlim(0, 60)
    axes[0].grid(True)
    axes[0].legend()
    
    # Graph 2: Seizure Brainwave Frequencies
    axes[1].plot(freqs_seiz, psd_seiz, color='red', label='Seizure (Label 1)')
    axes[1].set_title("Seizure Chunk Frequency Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim(0, 60)
    axes[1].grid(True)
    axes[1].legend()
    
    plt.suptitle("How the Fourier Transform Sees the Brainwaves")
    plt.tight_layout()
    plt.show()

def plot_feature_bars(X_tr, Y_tr):
    print("\n--- Generating Frequency Band Bar Chart ---")
    fs = 512
    
    normal_idx = np.where(Y_tr == 0)[0][0]
    seizure_idx = np.where(Y_tr == 1)[0][0]
    normal_wave = X_tr[normal_idx, 0, :]
    seizure_wave = X_tr[seizure_idx, 0, :]
    
    # Run Welch's method
    freqs_norm, psd_norm = welch(normal_wave, fs=fs, nperseg=512)
    freqs_seiz, psd_seiz = welch(seizure_wave, fs=fs, nperseg=512)
    
    def get_band_power(freqs, psd):
        delta = np.mean(psd[(freqs >= 0.5) & (freqs <= 4.0)])
        theta = np.mean(psd[(freqs > 4.0) & (freqs <= 8.0)])
        alpha = np.mean(psd[(freqs > 8.0) & (freqs <= 12.0)])
        beta  = np.mean(psd[(freqs > 12.0) & (freqs <= 30.0)])
        gamma = np.mean(psd[(freqs > 30.0) & (freqs <= 60.0)])
        return [delta, theta, alpha, beta, gamma]
        
    norm_bands = get_band_power(freqs_norm, psd_norm)
    seiz_bands = get_band_power(freqs_seiz, psd_seiz)
    
    labels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    x = np.arange(len(labels))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, norm_bands, width, label='Normal (Label 0)', color='green')
    ax.bar(x + width/2, seiz_bands, width, label='Seizure (Label 1)', color='red')

    ax.set_ylabel('Average Loudness / Power')
    ax.set_title('Frequency Band Comparison: What the AI Actually Sees')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

# --- THE EXECUTION FLOW ---

X_train, Y_train, X_val, Y_val, X_test, Y_test = data_loading()

plot_frequency_spectrum(X_train, Y_train)

plot_feature_bars(X_train, Y_train)

skl_pipeline(X_train, Y_train, X_val, Y_val, X_test, Y_test)