!pip install wfdb neurokit2
import wfdb
import numpy as np
import pandas as pd
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
import os

# Step 2: Generate synthetic data
def load_synthetic_data(duration=600, sampling_rate=700):
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=60, noise=0.05, random_state=42)
    np.random.seed(42)
    labels = np.random.choice([0, 1], size=len(ecg_signal), p=[0.6, 0.4])
    print(f"ECG shape: {ecg_signal.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    return ecg_signal, labels

ecg_signal, labels = load_synthetic_data()

# Step 3: Preprocess ECG
def preprocess_ecg(ecg_signal, labels, window_size=3500, sampling_rate=700):
    segments = []
    segment_labels = []
    for i in range(0, len(ecg_signal) - window_size, window_size // 2):  # Overlap windows
        segment = ecg_signal[i:i + window_size]
        window_labels = labels[i:i + window_size]
        # Assign label based on proportion of high stress (1)
        high_stress_proportion = np.mean(window_labels == 1)
        segment_label = 1 if high_stress_proportion > 0.4 else 0  # Threshold for balance
        segments.append(segment)
        segment_labels.append(segment_label)
    segments = np.array(segments)
    segment_labels = np.array(segment_labels)
    r_peaks_list = []
    for segment in segments:
        cleaned = nk.ecg_clean(segment, sampling_rate=sampling_rate)
        r_peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
        r_peaks_list.append(r_peaks)
    print(f"Number of segments: {len(segments)}")
    print(f"Segment shape: {segments.shape}")
    print(f"Label distribution: {np.bincount(segment_labels)}")
    return segments, r_peaks_list, segment_labels

segments, r_peaks_list, segment_labels = preprocess_ecg(ecg_signal, labels)

# Step 4: Extract HRV features
def extract_hrv_features(segments, r_peaks_list, sampling_rate=700):
    hrv_features = []
    for r_peaks in r_peaks_list:
        if len(r_peaks) >= 3:
            try:
                hrv_time = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)
                hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate, normalize=True)
                if hrv_freq.empty or hrv_time.empty:
                    hrv_features.append(np.zeros(20))
                else:
                    hrv = pd.concat([hrv_time, hrv_freq], axis=1)
                    hrv_features.append(hrv.iloc[0].values)
            except Exception as e:
                print(f"HRV computation failed for segment: {e}")
                hrv_features.append(np.zeros(20))
        else:
            hrv_features.append(np.zeros(20))
    hrv_features = np.array(hrv_features)
    hrv_features = np.nan_to_num(hrv_features, nan=0.0)
    scaler = StandardScaler()
    hrv_features = scaler.fit_transform(hrv_features)
    print(f"HRV features shape: {hrv_features.shape}")
    return hrv_features

hrv_features = extract_hrv_features(segments, r_peaks_list)

# Step 5: Train Random Forest
def train_rf_model(hrv_features, segment_labels):
    X_train, X_val, y_train, y_val = train_test_split(hrv_features, segment_labels, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    val_pred = rf_model.predict(X_val)
    print("Validation Set Evaluation:")
    print(classification_report(y_val, val_pred, target_names=['Low Stress', 'High Stress']))
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(rf_model.feature_importances_)), rf_model.feature_importances_)
    plt.title('Feature Importance in Random Forest')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()
    return rf_model, X_train, y_train, X_val, y_val

rf_model, X_train, y_train, X_val, y_val = train_rf_model(hrv_features, segment_labels)

# Step 6: Evaluate and save
def evaluate_and_save(rf_model, hrv_features, segment_labels):
    X_train_val, X_test, y_train_val, y_test = train_test_split(hrv_features, segment_labels, test_size=0.2, random_state=42)
    test_pred = rf_model.predict(X_test)
    print("Test Set Evaluation:")
    report = classification_report(y_test, test_pred, target_names=['Low Stress', 'High Stress'])
    print(report)
    os.makedirs('results', exist_ok=True)
    with open('results/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('results/classification_report.txt', 'w') as f:
        f.write("Test Set Evaluation:\n")
        f.write(report)
    return test_pred

test_pred = evaluate_and_save(rf_model, hrv_features, segment_labels)

# Download results
from google.colab import files
files.download('results/rf_model.pkl')
files.download('results/classification_report.txt')



















