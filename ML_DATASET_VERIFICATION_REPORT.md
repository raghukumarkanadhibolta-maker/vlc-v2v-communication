# VLC ML-Ready Dataset Verification Report
## Academic Project: AI-Based Adaptive Car-to-Car Visible Light Communication with ML-Based Channel Sensing

---

### Executive Summary

This report documents the rigorous pre-training verification of the newly generated ML-Ready Dataset:
* **Primary Target File**: `dataset/ml_ready/vlc_channel_ml_ready.csv`
* **Integrity Status**: **100% Unaltered (Read-Only Inspection)**
* **Model Training Status**: **Zero models trained**
* **Verification Status**: **PASSED ALL CHECKS**

---

## 1. File Specification & Storage Paths

* **Exact File Path (CSV)**: `c:\Users\raghu\OneDrive\Desktop\DVLC_VLC_ML_Animation\dataset\ml_ready\vlc_channel_ml_ready.csv`
* **File Size (CSV)**: **280,671 bytes** (~274.1 KB)
* **Exact File Path (Parquet)**: `c:\Users\raghu\OneDrive\Desktop\DVLC_VLC_ML_Animation\dataset\ml_ready\vlc_channel_ml_ready.parquet`
* **File Size (Parquet)**: **41,329 bytes** (~40.4 KB)

---

## 2. Dataset Dimensions

* **Total Number of Rows**: **6,924 rows** (100% preservation of all temporal sample windows)
* **Total Number of Columns**: **9 columns** (6 physical sensor inputs, 1 session splitting metadata, 2 ground-truth targets)
* **Total Matrix Elements**: $6,924 \times 9 = \mathbf{62,316\text{ values}}$

---

## 3. Feature Inventory & Data Types

| # | Column Name | Data Type | Functional Role | Physical Description |
| :-: | :--- | :-: | :--- | :--- |
| **1** | `session_id` | `int64` | Split Metadata | Experimental session index ($1$ to $12$) for GroupKFold cross-validation |
| **2** | `distancia_m` | `float64` | Physical Feature | Inter-vehicle line-of-sight distance in meters ($d$) |
| **3** | `ambientLight` | `float64` | Physical Feature | Background ambient optical illuminance in lux (lx) |
| **4** | `angRadiacion` | `float64` | Physical Feature | Transmitter LED radiation / irradiance angle ($\phi \in \{0^\circ, 30^\circ, 45^\circ\}$) |
| **5** | `angIncidencia` | `float64` | Physical Feature | Receiver photodiode incidence tilt angle ($\psi \in \{0^\circ, 30^\circ, 45^\circ\}$) |
| **6** | `presenciaPersonas` | `int64` | Physical Feature | Binary pedestrian presence / line-of-sight obstruction flag ($0$ or $1$) |
| **7** | `entorno_encoded` | `int64` | Physical Feature | Binary encoded test environment ($0$ = `laboratorio`, $1$ = `pasilloCristal`) |
| **8** | `target_channel_quality_binary` | `int64` | Primary Target | Binary channel state ($1$ = Good Channel, $0$ = Poor Channel @ $-35.0$ dB) |
| **9** | `target_channel_quality_multi` | `int64` | Multi-Class Target | 3-class state ($2$ = Good, $1$ = Moderate, $0$ = Poor @ $-34$ dB / $-38$ dB) |

---

## 4. Target Columns & Class Distribution Analysis

### 4.1 Primary Binary Target (`target_channel_quality_binary`):
* **Established Physical Cutoff**: $\text{Path Loss} \ge -35.0\text{ dB}$ (Empirical Median Split)
* **Class 0 (Poor Channel)**: **3,591 samples (51.86%)**
* **Class 1 (Good Channel)**: **3,333 samples (48.14%)**
* **Class Ratio (Poor / Good)**: **$1.0774 : 1$**
* **Balance Assessment**: **PERFECTLY BALANCED**. Shannon target entropy $H(Y) = 0.9994\text{ bits}$ ($\approx 1.0$ bit). Eliminates majority-class distortion and removes any requirement for synthetic resampling (SMOTE) or class re-weighting.

### 4.2 Multi-Class Target (`target_channel_quality_multi`):
* **Established Physical Cutoffs**: Class 2 ($\ge -34.0\text{ dB}$), Class 1 ($-38.0\text{ to } -34.0\text{ dB}$), Class 0 ($< -38.0\text{ dB}$, Otsu/K-means boundary).
* **Class 2 (Good / High-Throughput)**: **2,348 samples (33.91%)**
* **Class 1 (Moderate / Safe Margin)**: **3,003 samples (43.37%)**
* **Class 0 (Poor / Deep Fade / Blockage)**: **1,573 samples (22.72%)**
* **Balance Assessment**: **WELL-DISTRIBUTED**. All three physical communication states are strongly populated; the minority class represents over $22.7\%$ of the data.

---

## 5. Data Quality Audit: Missing Values & Duplicate Rows

* **Missing Values (`NaN` / `null`)**: Exactly **`0` nulls across all 9 columns (0.00%)**. Complete data density.
* **Infinite Values (`±inf`)**: Exactly **`0` infinite values (0.00%)**.
* **Duplicate Rows across all 9 columns**: **1,814 duplicate rows (26.20%)**.
  * *Physical Origin*: The hardware platform recorded repeated temporal sampling windows while maintaining static mechanical stage configurations (fixed distance, discrete angles, constant ambient light). Because the micro-fluctuating photodiode voltage and millisecond timestamps were excluded to prevent target leakage, consecutive windows within the same static configuration naturally exhibit identical feature vectors.
  * *ML Guidance*: These reflect true physical experimental duration (dwell time) at specific channel conditions. GroupKFold splitting by `session_id` ensures duplicates within the same experimental run never bridge train and test folds.

---

## 6. Summary Statistics of Numerical Features ($N = 6,924$)

| Feature Name | Minimum | Maximum | Mean | Standard Deviation | Physical Scale / Units |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `session_id` | $1.0000$ | $12.0000$ | $6.2695$ | $3.4269$ | Discrete integer runs ($1 - 12$) |
| `distancia_m` | $0.0070$ | $2.3660$ | $1.2012$ | $0.4461$ | Inter-vehicle distance in meters |
| `ambientLight` | $1.9190$ | $1,984.0820$ | $115.9727$ | $298.8788$ | Background illuminance in lux |
| `angRadiacion` | $0.0000$ | $45.0000$ | $24.1270$ | $18.6687$ | Transmitter LED angle in degrees |
| `angIncidencia` | $0.0000$ | $45.0000$ | $12.3115$ | $17.8089$ | Receiver photodiode angle in degrees |
| `presenciaPersonas` | $0.0000$ | $1.0000$ | $0.4555$ | $0.4981$ | Binary pedestrian blockage ($0/1$) |
| `entorno_encoded` | $0.0000$ | $1.0000$ | $0.0729$ | $0.2600$ | Binary environment encoding ($0/1$) |
| `target_channel_quality_binary` | $0.0000$ | $1.0000$ | $0.4814$ | $0.4997$ | Binary channel quality label ($0/1$) |
| `target_channel_quality_multi` | $0.0000$ | $2.0000$ | $1.1119$ | $0.7442$ | Multi-class channel label ($0/1/2$) |

---

## 7. Target Leakage Confirmation

An exhaustive algorithmic scan was executed across all column headers searching for prohibited signal-reception terms:

| Prohibited Variable Group | Checked Terms | Present in Feature Matrix $\mathbf{X}$? | Audit Result |
| :--- | :--- | :---: | :---: |
| **Path Loss / Attenuation** | `pathloss`, `pathloss_clean_db` | **NO** | **PASSED** |
| **Voltage Amplitude** | `vpp_volts`, `vpp_counts` | **NO** | **PASSED** |
| **Received Optical Power** | `p_rx_uW`, `p_rx_dBm` | **NO** | **PASSED** |
| **Incident Ambient Power** | `p_amb_uW` | **NO** | **PASSED** |
| **Signal-to-Noise Ratio** | `snr_db`, `snr_linear` | **NO** | **PASSED** |
| **Bit Error Rate** | `theoretical_ber_ook`, `log10_ber` | **NO** | **PASSED** |
| **Outage Indicators** | `is_pathloss_outage` | **NO** | **PASSED** |
| **Photocurrents** | `i_sig_A`, `i_bg_A` | **NO** | **PASSED** |

> [!IMPORTANT]
> **CONFIRMED**: Zero received-signal-derived variables exist in the feature matrix $\mathbf{X}$. The dataset is 100% free of target leakage.

---

## 8. Encoding Verification: `entorno_encoded`

Cross-tabulation against the ground-truth string environment (`entorno`) from the preprocessed dataset:

$$\begin{array}{c|cc|c}
\text{Raw Environment (\texttt{entorno})} & \texttt{entorno\_encoded = 0} & \texttt{entorno\_encoded = 1} & \text{Total} \\
\hline
\text{\texttt{laboratorio}} & 6,419 & 0 & 6,419 \\
\text{\texttt{pasilloCristal}} & 0 & 505 & 505 \\
\hline
\text{Total} & 6,419 & 505 & 6,924
\end{array}$$

* **Verification Result**: **100% Correct**.
  * `laboratorio` $\to$ **`0`** ($6,419$ samples, $92.71\%$)
  * `pasilloCristal` $\to$ **`1`** ($505$ samples, $7.29\%$)
  * Zero misclassifications, zero orphan categories.

---

## 9. Pre-Training Readiness Verdict

| Audit Item | Expected Criteria | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **File Existence & Format** | CSV and Parquet in `dataset/ml_ready/` | Present, valid syntax | **PASS** |
| **Dimensionality** | 6,924 samples $\times$ 9 features | Exactly 6,924 $\times$ 9 | **PASS** |
| **Target Leakage** | 0 received-signal features in $\mathbf{X}$ | Exactly 0 found | **PASS** |
| **Input Feature Scope** | Physical vehicle sensor inputs only | distancia, lux, angles, personas, entorno | **PASS** |
| **Missing Data** | 0 nulls | 0 nulls across all cells | **PASS** |
| **Encoding Integrity** | Clean binary map for `entorno` | 100% verified cross-tabulation | **PASS** |
| **Target Balance** | Balanced binary distribution | 51.86% vs 48.14% ($1.08:1$) | **PASS** |
| **Model Training** | Zero models trained | Zero models trained | **PASS** |

**FINAL VERDICT**: The dataset [`vlc_channel_ml_ready.csv`](file:///c:/Users/raghu/OneDrive/Desktop/DVLC_VLC_ML_Animation/dataset/ml_ready/vlc_channel_ml_ready.csv) is **fully verified and certified ML-ready** for supervised model training.
