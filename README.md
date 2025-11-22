# SAND_IEEE_COMPETTION
# Dysarthria Detection with Librosa Features and Optimized XGBoost

> SAND IEEE Competition – ICASSP 2026 Dysarthria Detection Challenge Submission

This repository contains our **fully reproducible dysarthria detection system** designed for the ICASSP 2026 Grand Challenge.  
The key constraint of the challenge is that **Praat/Parselmouth are forbidden** and the solution must run using **Kaggle-only dependencies**.

We address this by building a **Librosa + NumPy–only acoustic feature pipeline**, integrating **class rebalancing**, **Optuna-tuned XGBoost**, and **probability calibration with confidence-based refinement**.  
Our final system achieves a **weighted-F1 score of 0.65** on the official validation set. :contentReference[oaicite:1]{index=1}

---

## 🔍 Problem & Breakthrough

**Problem.** Dysarthria affects phonation, articulation, and speech rhythm. The challenge is to detect **5 levels of dysarthria severity (0–4)** from a small, imbalanced dataset, **without using Praat-based tools** and within Kaggle’s restricted environment. :contentReference[oaicite:2]{index=2}  

**Our breakthrough:**

1. **Praat-free clinical voice descriptors**  
   We re-create key clinical voice biomarkers (e.g., jitter, shimmer, HNR, CPP, formants) using **only Librosa + NumPy**, staying within Kaggle constraints while preserving interpretability and clinical relevance. :contentReference[oaicite:3]{index=3}  

2. **Multi-task acoustic fusion across 8 phonation/rhythm tasks**  
   Each speaker performs:
   - 5 sustained vowels: `/A/`, `/E/`, `/I/`, `/O/`, `/U/`
   - 3 rhythmic sequences: `PA`, `TA`, `KA`  
   We extract features per task and then **fuse them at subject level**, producing a rich representation (≈180–220 features per subject). :contentReference[oaicite:4]{index=4}  

3. **SMOTE–Tomek hybrid balancing**  
   To combat label imbalance, we use a **SMOTE–Tomek** pipeline that simultaneously oversamples minority classes and cleans ambiguous samples along class boundaries. :contentReference[oaicite:5]{index=5}  

4. **Optuna-optimized XGBoost**  
   We treat dysarthria severity prediction as a **5-class classification** problem using `multi:softprob` XGBoost, with **hyperparameters tuned via Optuna** over a carefully chosen search space. :contentReference[oaicite:6]{index=6}  

5. **Probability calibration + confidence thresholding**  
   We post-process XGBoost outputs with:
   - **Sigmoid probability calibration** to avoid overconfident predictions.
   - A **confidence threshold (τ = 0.42)**: if the top class probability is below τ, we fall back to the **second-most likely class** to reduce confusing overlaps. :contentReference[oaicite:7]{index=7}  

Together, these components form a **lightweight, interpretable, and competition-compliant system** that performs competitively against more complex baselines.

---

## 📂 Dataset & Labels

- **Tasks per speaker:**  
  - 5 sustained vowels: `/A/`, `/E/`, `/I/`, `/O/`, `/U/`  
  - 3 rhythmic tasks: `PA`, `TA`, `KA`
- **Audio preprocessing:**
  - All recordings resampled to **16 kHz** and amplitude-normalized.
  - Missing recordings are handled with **binary indicators** so that feature dimensionality stays fixed. :contentReference[oaicite:8]{index=8}  
- **Labels:**
  - Dysarthria severity: **5 classes (0–4)**.
- **Official validation set** is used for reporting final metrics.

---

## 🎛 Feature Extraction (Librosa + NumPy Only)

All features are created with **Librosa** and **NumPy**, strictly avoiding Praat/parselmouth. :contentReference[oaicite:9]{index=9}  

### 1. Cepstral Features (MFCCs)

- 13 MFCC coefficients per recording.
- For each, we compute **mean and standard deviation** → **26-dimensional vector per task**.

### 2. LPC-Based Formants

- Estimate resonances (F1–F3) from **LPC polynomial roots**.
- Keep roots corresponding to resonances **90–5000 Hz** to approximate **first three formants**. :contentReference[oaicite:10]{index=10}  

### 3. Voice Quality Measures

Computed per recording:

- **Jitter** – variability in pitch period, approximated via zero-crossing-based pitch tracking.
- **Shimmer** – variability in amplitude (RMS-based).
- **HNR (Harmonic-to-Noise Ratio)** – derived via harmonic–percussive source separation (HPSS).
- **CPP (Cepstral Peak Prominence)** – approximated from log-magnitude STFT and cepstral peak height. :contentReference[oaicite:11]{index=11}  

### 4. Energy Descriptors

- Short-term energy **mean** and **variance** to capture phonatory stability.

### 5. Rhythmic Features for PA/TA/KA

- Onset-based measures for rhythmic tasks:
  - Voice Onset Time (VOT) statistics: **mean, variance, count, max interval**.
- Provides rhythm/coordination information across `PA`, `TA`, `KA`. :contentReference[oaicite:12]{index=12}  

### 6. Subject-Level Fusion

- All per-task features (vowels + rhythmic tasks) are **concatenated** to form a **single subject vector** (~180–220 features).
- This models the speaker holistically instead of treating each file independently. :contentReference[oaicite:13]{index=13}  

---

## 🧠 Classification Pipeline

### 1. Scaling & Class Balancing

- **StandardScaler** for feature normalization.
- **SMOTE–Tomek** to:
  - Oversample minority classes,
  - Remove borderline samples via Tomek links. :contentReference[oaicite:14]{index=14}  

### 2. XGBoost Classifier

- Objective: `multi:softprob` (5-way classification).
- Initial configuration:
  - `n_estimators = 300`
  - `learning_rate = 0.05`
  - `max_depth = 6`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8` :contentReference[oaicite:15]{index=15}  

### 3. Optuna Hyperparameter Optimization

**Search space** (50 trials): :contentReference[oaicite:16]{index=16}  

- `lambda`, `alpha` ∈ [1e-3, 10]
- `max_depth` ∈ [3, 10]
- `min_child_weight` ∈ [1, 10]
- `eta` ∈ [0.01, 0.3]
- `gamma` ∈ [0, 10]
- `subsample`, `colsample_bytree` ∈ [0.4, 1]

**Best hyperparameters** selected by Optuna: :contentReference[oaicite:17]{index=17}  

- `max_depth = 7`
- `eta = 0.054`
- `subsample = 0.81`
- `colsample_bytree = 0.74`
- `lambda = 0.018`
- `alpha = 0.11`
- `gamma = 1.9`
- `min_child_weight = 3`

### 4. Calibration & Confidence Thresholding

- Apply **sigmoid calibration** on XGBoost probabilities.
- Introduce a **confidence threshold** τ:

  > If the maximum class probability < τ,  
  > replace the prediction with the **second-highest** probability class.

- Optimal threshold: **τ = 0.42**. :contentReference[oaicite:18]{index=18}  

This makes predictions **less overconfident** and improves robustness in ambiguous regions, even if the overall weighted-F1 stays similar.

---

## 📊 Results

### Cross-Validation (Training Data)

Using **4-fold Stratified Cross-Validation** on the SMOTE–Tomek balanced training data: :contentReference[oaicite:19]{index=19}  

- **Weighted-F1:** 0.65  
- **Accuracy:** 0.65  

### Official Validation Set

Challenge metric: **Weighted-F1**

| Model Variant                     | Weighted-F1 |
|-----------------------------------|-------------|
| XGBoost (Optuna-best)            | 0.65        |
| + Calibration                     | 0.65        |
| + Confidence Thresholding (τ=0.42)| 0.65        |

The calibrated + thresholded model shows **more stable and interpretable confidence** without harming the main competition metric. :contentReference[oaicite:20]{index=20}  

---

## 🧪 How to Run (Typical Workflow)

> Adjust paths / filenames to match your repository structure.

1. **Install dependencies**

   ```bash
   pip install numpy librosa scikit-learn xgboost imbalanced-learn optuna
