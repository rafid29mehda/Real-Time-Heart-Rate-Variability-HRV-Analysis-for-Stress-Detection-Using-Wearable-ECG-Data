# Real-Time Heart Rate Variability (HRV) Analysis for Stress Detection

## Project Overview
This project implements a pipeline for real-time stress detection using Heart Rate Variability (HRV) features extracted from synthetic electrocardiogram (ECG) data. The goal is to develop a lightweight machine learning model suitable for wearable devices, such as smartwatches, to classify stress levels (low vs. high). The pipeline processes ECG signals, extracts HRV features, and trains a Random Forest classifier to predict stress, demonstrating expertise in biomedical signal processing and machine learning. This project complements other portfolio projects (e.g., ECG arrhythmia classification) by focusing on feature engineering and real-time analysis, enhancing the author’s PhD profile in Biomedical Signal Processing.

## Objectives
- Generate synthetic ECG data to simulate wearable device inputs.
- Segment ECG signals into overlapping windows for real-time processing.
- Extract time- and frequency-domain HRV features (e.g., RMSSD, SDNN, LF/HF ratio).
- Train and evaluate a Random Forest classifier for binary stress classification.
- Organize the codebase in a modular, professional structure for GitHub.

## Dataset
- **Source**: Synthetic ECG data generated using the `neurokit2` library.
- **Specifications**:
  - Duration: 10 minutes (600 seconds).
  - Sampling Rate: 700 Hz, producing 420,000 samples.
  - Heart Rate: 60 beats per minute (BPM) with 5% noise for realism.
- **Labels**: Binary stress labels (0: low stress, 1: high stress) randomly assigned with a 60:40 probability (60% low stress, 40% high stress).
- **Rationale**: Synthetic data is used due to restricted access to real datasets like WESAD. The synthetic ECG mimics real-world signals, allowing the pipeline to be tested and extended to real data in future work.

## Methodology
The project follows a six-step pipeline, implemented in Python and designed to run in Google Colab. Each step is encapsulated in a modular function, stored in a separate `.py` file within the `src/` folder.

### Step 1: Install Dependencies
- **Purpose**: Sets up the Python environment with required libraries.
- **Libraries**:
  - `wfdb`: For reading ECG data (included for potential real dataset use).
  - `numpy`, `pandas`: For numerical computations and data manipulation.
  - `scikit-learn`: For preprocessing, Random Forest training, and evaluation.
  - `neurokit2`: For ECG signal generation, cleaning, R-peak detection, and HRV feature extraction.
  - `matplotlib`: For visualizing feature importance and results.
- **Implementation**: Uses `pip install` to ensure libraries are available in Colab and imports them for use.
- **Output**: Confirmation of successful installation (e.g., `Successfully installed ...`).

### Step 2: Generate Synthetic Data (`load_synthetic_data`)
- **Purpose**: Creates a synthetic ECG signal and corresponding stress labels to simulate wearable device data.
- **Function**: `load_synthetic_data(duration=600, sampling_rate=700)`
  - Generates a 10-minute ECG signal using `nk.ecg_simulate` with a heart rate of 60 BPM and 5% noise for realism.
  - Assigns binary labels (0: low stress, 1: high stress) with a 60:40 probability using `np.random.choice`.
  - Prints the shapes of the ECG signal and labels, and the label distribution.

- **Rationale**: Synthetic data ensures reproducibility and accessibility, mimicking real ECG signals for testing.

### Step 3: Preprocess ECG (`preprocess_ecg`)
- **Purpose**: Segments the ECG signal into overlapping windows and detects R-peaks for HRV analysis.
- **Function**: `preprocess_ecg(ecg_signal, labels, window_size=3500, sampling_rate=700)`
- Segments the ECG into 5-second windows (3500 samples at 700 Hz) with 50% overlap (step size: `window_size // 2`) to capture temporal dynamics.
- Assigns a binary label to each segment based on the proportion of high-stress labels (1) in the window:
  - If the proportion exceeds 0.4, the segment is labeled as high stress (1); otherwise, low stress (0).
  - This threshold ensures a balanced label distribution, addressing previous issues with single-class segments.
- Cleans each segment using `nk.ecg_clean` and detects R-peaks with `nk.ecg_peaks`.
- Returns an array of segments, a list of R-peak indices, and segment labels.
