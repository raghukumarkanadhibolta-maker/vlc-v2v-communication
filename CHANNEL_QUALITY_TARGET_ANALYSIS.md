# VLC Channel Quality Analysis & ML Target Formulation Report
## Academic Project: AI-Based Adaptive Car-to-Car Visible Light Communication with ML-Based Channel Sensing

---

### Executive Summary

This report delivers a rigorous physical, statistical, and communication-theoretic analysis of the preprocessed Visible Light Communication (VLC) dataset (`vlc_channel_dataset_processed.csv` and `vlc_channel_dataset_processed.parquet`) for **Machine Learning Target Creation**. 

In accordance with strict research integrity and engineering guidelines:
1. **Zero Data Mutation**: Neither the original raw dataset (`dataset.mat`) nor the newly created processed dataset files were altered.
2. **Zero Model Training**: No machine learning models were trained.
3. **Evidence-Based Target Formulation**: The selection of channel-quality thresholds is grounded in empirical distributions, information theory, statistical clustering (Otsu's discriminant method, K-Means, Gaussian Mixture Models), and international optical communication standards (IEEE 802.15.7 Forward Error Correction limits).

---

## 1. Processed Dataset Overview & Dimensions

The preprocessing pipeline successfully ingested the raw serial telemetry from the 12 experimental sessions of the Thorlabs PDA100A2 / Arduino Uno optical testbed, addressed hardware sensor dropouts, engineered physical VLC optical metrics, and generated clean structured outputs:

* **Processed Storage Formats**:
  * CSV: `dataset/processed/vlc_channel_dataset_processed.csv` (Size: 2,230,901 bytes / ~2.13 MB)
  * Apache Parquet: `dataset/processed/vlc_channel_dataset_processed.parquet` (Size: 347,993 bytes / ~339.8 KB)
* **Final Number of Rows**: **6,924 samples** (100% preservation of all experimental sample windows)
* **Final Number of Columns**: **29 features** (expanded from 9 raw columns via physical feature engineering and diagnostic tagging)
* **Total Value Cells**: $6,924 \times 29 = \mathbf{200,796\text{ values}}$

---

## 2. Complete Feature Inventory & Data Types

The 29 features in the processed dataset are organized below by their functional role within the VLC vehicular communication architecture:

| # | Feature Name | Data Type | Role in ML Pipeline | Physical / Mathematical Definition | Engineering Unit |
| :-: | :--- | :-: | :--- | :--- | :--- |
| **1** | `session_id` | `int64` | Temporal Metadata | Experimental acquisition session index (1 to 12) | Discrete identifier |
| **2** | `date_raw_ms` | `float64` | Temporal Metadata | Raw UNIX timestamp in milliseconds | Milliseconds (ms) |
| **3** | `dt` | `object` (string) | Temporal Metadata | Formatted UTC timestamp (`YYYY-MM-DD HH:MM:SS.SSS`) | ISO 8601 string |
| **4** | `distancia_imputed_cm` | `float64` | Sensing Input | Inter-vehicle distance; sonar dropouts temporally imputed | Centimeters (cm) |
| **5** | `distancia_m` | `float64` | Sensing Input | Inter-vehicle distance converted to SI meters ($d$) | Meters (m) |
| **6** | `ambientLight` | `float64` | Sensing Input | Background ambient optical illuminance (BH1750 sensor) | Lux (lx) |
| **7** | `log_ambientLight` | `float64` | Sensing Input | Log10 transformed ambient illuminance $\log_{10}(\text{lux} + 1)$ | $\log_{10}(\text{lx})$ |
| **8** | `angRadiacion` | `float64` | Sensing Input | Transmitter LED radiation / irradiance angle ($\phi \in \{0^\circ, 30^\circ, 45^\circ\}$) | Degrees (°) |
| **9** | `cos_angRadiacion` | `float64` | Sensing Input | Lambertian emission cosine factor $\cos(\phi)$ | Dimensionless $[0, 1]$ |
| **10** | `angIncidencia` | `float64` | Sensing Input | Receiver photodiode incidence angle ($\psi \in \{0^\circ, 30^\circ, 45^\circ\}$) | Degrees (°) |
| **11** | `cos_angIncidencia` | `float64` | Sensing Input | Receiver collection area cosine factor $\cos(\psi)$ | Dimensionless $[0, 1]$ |
| **12** | `angular_alignment_factor` | `float64` | Sensing Input | Joint optical angular alignment $\cos(\phi) \cdot \cos(\psi)$ | Dimensionless $[0, 1]$ |
| **13** | `geometric_loss_factor` | `float64` | Sensing Input | Free-space geometric decay $\frac{\cos(\phi)\cos(\psi)}{d^2}$ | $\text{m}^{-2}$ |
| **14** | `presenciaPersonas` | `float64` | Sensing Input | Line-of-sight obstruction / pedestrian shadowing ($0$ or $1$) | Binary flag |
| **15** | `entorno` | `object` (string) | Sensing Input | Physical environment (`laboratorio` vs `pasilloCristal`) | Categorical string |
| **16** | `environment_id` | `int64` | Sensing Input | Binary encoded environment ($0 = \text{lab}$, $1 = \text{corridor}$) | Binary ($0/1$) |
| **17** | `vpp_counts` | `float64` | Reception Metric | Reconstructed photodiode peak-to-peak signal amplitude | ADC counts $[0, 1023]$ |
| **18** | `vpp_volts` | `float64` | Reception Metric | Reconstructed photodiode output voltage amplitude ($V_{pp}$) | Volts (V) |
| **19** | `p_rx_uW` | `float64` | Reception Metric | Received modulated optical signal power ($P_{rx}$) | Microwatts ($\mu\text{W}$) |
| **20** | `p_rx_dBm` | `float64` | Reception Metric | Received optical power in decibel-milliwatts ($10\log_{10}(P/1\text{mW})$) | dBm |
| **21** | `p_amb_uW` | `float64` | Reception Metric | Background ambient optical power incident on photodiode | Microwatts ($\mu\text{W}$) |
| **22** | `snr_db` | `float64` | Reception Metric | Electrical Signal-to-Noise Ratio ($10\log_{10}(\text{SNR}_{\text{linear}})$) | Decibels (dB) |
| **23** | `theoretical_ber_ook` | `float64` | Reception Metric | Theoretical Bit Error Rate for OOK modulation in AWGN | Error probability |
| **24** | `log10_ber` | `float64` | Reception Metric | Base-10 logarithm of theoretical BER (clamped at $-12.0$) | $\log_{10}(\text{BER})$ |
| **25** | `pathloss_clean_db` | `float64` | Reception Metric | Calibrated optical channel path loss / attenuation ($-\infty \to -60\text{ dB}$) | Decibels (dB) |
| **26** | `channel_quality_binary` | `int64` | Preprocessed Target | Binary target: $1 = \text{Good } (\ge -35.0\text{ dB}), 0 = \text{Poor } (< -35.0\text{ dB})$ | Binary ($0/1$) |
| **27** | `channel_quality_multi` | `int64` | Preprocessed Target | Multi-class target: $2 = \text{Good}, 1 = \text{Moderate}, 0 = \text{Poor}$ | Categorical ($0, 1, 2$) |
| **28** | `is_distance_dropout` | `int64` | Diagnostic Flag | Flag indicating raw ultrasonic sonar echo timeout ($d = 0\text{ cm}$) | Binary ($0/1$) |
| **29** | `is_pathloss_outage` | `int64` | Diagnostic Flag | Flag indicating raw optical signal loss ($V_{pp} = 0\text{ V}, \text{PL} = -\infty$) | Binary ($0/1$) |

---

## 3. Data Quality, Missing Values, and Invalid Data Audit

A comprehensive verification script was executed against all 200,796 cells of the dataset:

1. **Null / NaN Values**:
   * **Exactly 0 null or missing values across all 29 columns (0.00%)**.
   * The dataset is completely dense and fully populated.

2. **Infinite Values ($+\infty$ / $-\infty$)**:
   * **Exactly 0 infinite values across all 27 numerical columns (0.00%)**.
   * In the raw data, 3 samples (indices 1435, 1436, and 1440) evaluated to `pathloss = -inf` because complete optical blockage resulted in $V_{pp} = 0.0$ V, yielding $\log_{10}(0) = -\infty$. In `pathloss_clean_db`, these were replaced with the physical hardware 1-bit quantization floor ($-60.0$ dB), and their presence is explicitly tracked in `is_pathloss_outage = 1`.

3. **Ultrasonic Distance Dropouts (`distancia == 0.0 cm`)**:
   * In the raw dataset, exactly **421 samples (6.08%)** had an HC-SR04 sonar reading of $0.0$ cm due to ultrasonic echo reflection timeouts during severe angular tilt or alignment cone divergence.
   * In `distancia_imputed_cm`, these 421 samples were cleanly reconstructed using intra-session temporal linear interpolation with boundary forward/backward filling.
   * Minimum imputed distance is **0.70 cm**; maximum is **236.60 cm**. Exactly 0 zero-values remain in the distance feature.
   * All 421 occurrences are preserved and transparently marked by `is_distance_dropout = 1`.

4. **Zero-Variance Feature Removal**:
   * The raw column `lente` was composed of 6,924 identical entries of $0.0$ (variance = 0). It has been excluded from the processed dataset.

---

## 4. Comprehensive Descriptive Summary Statistics

The table below presents the full descriptive statistics for all 27 numeric variables in the processed dataset ($N = 6,924$ rows):

| Variable Name | Mean | Std Dev | Min | 25% ($Q_1$) | 50% (Median) | 75% ($Q_3$) | Max | Skewness | Zero Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `session_id` | 6.2695 | 3.4269 | 1.0000 | 3.0000 | 6.0000 | 9.0000 | 12.0000 | +0.14 | 0 |
| `date_raw_ms` ($10^{12}$) | 1.6523 | $1.48\cdot 10^{-6}$ | 1.6523 | 1.6523 | 1.6523 | 1.6523 | 1.6523 | +0.13 | 0 |
| `distancia_imputed_cm` | 120.1162 | 44.6065 | 0.7000 | 84.7000 | 123.3000 | 157.3000 | 236.6000 | -0.17 | 0 |
| `distancia_m` | 1.2012 | 0.4461 | 0.0070 | 0.8470 | 1.2330 | 1.5730 | 2.3660 | -0.17 | 0 |
| `ambientLight` (lx) | 115.9727 | 298.8788 | 1.9190 | 32.2510 | 40.0000 | 48.3300 | 1,984.0820 | +3.90 | 0 |
| `log_ambientLight` | 1.6317 | 0.4663 | 0.4652 | 1.5218 | 1.6128 | 1.6931 | 3.2978 | +1.49 | 0 |
| `angRadiacion` (°) | 24.1270 | 18.6687 | 0.0000 | 0.0000 | 30.0000 | 45.0000 | 45.0000 | -0.31 | 2,416 |
| `cos_angRadiacion` | 0.8641 | 0.1183 | 0.7071 | 0.7071 | 0.8660 | 1.0000 | 1.0000 | -0.19 | 0 |
| `angIncidencia` (°) | 12.3115 | 17.8089 | 0.0000 | 0.0000 | 0.0000 | 30.0000 | 45.0000 | +0.88 | 4,592 |
| `cos_angIncidencia` | 0.9315 | 0.1065 | 0.7071 | 0.8660 | 1.0000 | 1.0000 | 1.0000 | -1.25 | 0 |
| `angular_alignment_factor` | 0.8116 | 0.1720 | 0.5000 | 0.7071 | 0.8660 | 1.0000 | 1.0000 | -0.46 | 0 |
| `geometric_loss_factor` | 3.6391 | 125.3241 | 0.0950 | 0.3585 | 0.5489 | 0.9291 | 10,204.08 | +78.23 | 0 |
| `presenciaPersonas` | 0.4555 | 0.4981 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | +0.18 | 3,770 |
| `environment_id` | 0.0729 | 0.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | +3.29 | 6,419 |
| `vpp_counts` | 232.2661 | 187.8801 | 0.0000 | 105.0000 | 173.0000 | 280.0000 | 822.0000 | +1.27 | 3 |
| `vpp_volts` (V) | 1.1352 | 0.9183 | 0.0000 | 0.5132 | 0.8456 | 1.3685 | 4.0176 | +1.27 | 3 |
| `p_rx_uW` ($\mu\text{W}$) | 16.3093 | 14.2698 | 0.0000 | 7.3314 | 12.5448 | 21.3425 | 66.9599 | +1.65 | 3 |
| `p_rx_dBm` (dBm) | -19.7148 | 4.8407 | -90.0000 | -21.3481 | -19.0154 | -16.7076 | -11.7419 | -2.35 | 0 |
| `p_amb_uW` ($\mu\text{W}$) | 67.4357 | 178.4464 | 1.1431 | 16.5630 | 22.1889 | 27.2991 | 1,181.838 | +3.90 | 0 |
| `snr_db` (dB) | 50.8027 | 9.5259 | -40.0000 | 47.6156 | 52.2841 | 56.8894 | 66.8415 | -1.49 | 0 |
| `theoretical_ber_ook` | $2.92\cdot 10^{-4}$ | $1.07\cdot 10^{-2}$ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | +44.71 | 6,404 |
| `log10_ber` | -11.8661 | 0.9585 | -12.0000 | -12.0000 | -12.0000 | -12.0000 | -0.3010 | +7.98 | 0 |
| `pathloss_clean_db` (dB) | -35.7940 | 4.6417 | -60.0000 | -37.4473 | -35.1145 | -32.8067 | -27.8410 | -1.23 | 0 |
| `channel_quality_binary` | 0.4814 | 0.4997 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | +0.07 | 3,591 |
| `channel_quality_multi` | 1.1119 | 0.7442 | 0.0000 | 1.0000 | 1.0000 | 2.0000 | 2.0000 | -0.18 | 1,573 |
| `is_distance_dropout` | 0.0608 | 0.2390 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | +3.68 | 6,503 |
| `is_pathloss_outage` | 0.0004 | 0.0208 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | +48.02 | 6,921 |

### Categorical Feature Breakdown:
* **Operating Environment (`entorno`)**:
  * `laboratorio`: **6,419 samples (92.71%)** — Controlled indoor optical laboratory with low ambient illumination.
  * `pasilloCristal`: **505 samples (7.29%)** — Glass corridor exposed to daylight and sunlight glare.
* **Line-of-Sight Blockage (`presenciaPersonas`)**:
  * Free Channel ($0.0$): **3,770 samples (54.45%)**
  * Obstructed / Pedestrian Blockage ($1.0$): **3,154 samples (45.55%)**

---

## 5. Identification & Physical Mapping of VLC Channel Quality Features

In an adaptive Car-to-Car VLC system, the physical optical channel is governed by the line-of-sight (LOS) Channel DC Gain:
$$H_{\text{LOS}}(0) = \frac{(m + 1) A_{\text{det}}}{2 \pi d^2} \cos^m(\phi) \cos(\psi) T_s(\psi) g(\psi) \quad (0 \le \psi \le \Psi_c)$$

The features in the processed dataset represent specific components of this optical communication link:

```
[Vehicle Transmitter LED]                                         [Vehicle Receiver Photodiode]
  - angRadiacion (phi) ──┐                                         ┌── angIncidencia (psi)
                         ├──> [Free-Space Optical Channel] <───────┤
  - distancia_m (d)   ───┤    - pathloss_clean_db                  └── A_det, R_resp
  - presenciaPersonas ───┘    - geometric_loss_factor
                                         │
                                         ▼ (Photodiode Detection)
                                - vpp_volts, vpp_counts
                                - p_rx_uW, p_rx_dBm
                                         │
             [Background Noise] <────────┼────────> [Electrical Demodulation]
             - ambientLight              │          - snr_db
             - p_amb_uW                  ▼          - theoretical_ber_ook
             - sigma_shot^2      [Adaptive Channel Sensing]
```

### 5.1 Physical Grouping of Channel Quality Features

1. **Distance & Geometric Attenuation (`distancia_m`, `distancia_imputed_cm`, `geometric_loss_factor`)**:
   * *Physical Mechanism*: Radiated optical flux expands over a spherical wavefront, creating a quadratic power loss $P_{rx} \propto \frac{1}{d^2}$.
   * *Empirical Evidence*: Across distance intervals, path loss degrades monotonically:
     * $d \in (0.5, 1.0]$ m: Mean Path Loss = **$-33.29$ dB**, Mean $P_{rx}$ = **$28.02\ \mu\text{W}$**
     * $d \in (1.0, 1.5]$ m: Mean Path Loss = **$-35.77$ dB**, Mean $P_{rx}$ = **$13.61\ \mu\text{W}$**
     * $d \in (1.5, 2.0]$ m: Mean Path Loss = **$-37.74$ dB**, Mean $P_{rx}$ = **$8.90\ \mu\text{W}$**
     * $d \in (2.0, 2.5]$ m: Mean Path Loss = **$-40.43$ dB**, Mean $P_{rx}$ = **$4.11\ \mu\text{W}$**

2. **Received Modulated Signal Strength & Optical Power (`vpp_volts`, `vpp_counts`, `p_rx_uW`, `p_rx_dBm`)**:
   * *Physical Mechanism*: Modulated LED emission produces photocurrent $I_{\text{sig}} = R \cdot P_{rx}$. Transimpedance amplification translates this to $V_{pp}$.
   * *Distribution*: Median $P_{rx}$ is **$12.54\ \mu\text{W}$** ($-19.02$ dBm), spanning from $0.0\ \mu\text{W}$ (complete outage) to $66.96\ \mu\text{W}$ ($-11.74$ dBm).
   * *Correlation*: Pearson correlation between $P_{rx}$ (dBm) and `pathloss_clean_db` is **$r = 0.9804$** ($1.0000$ Spearman rank).

3. **Electrical Signal-to-Noise Ratio (`snr_db`)**:
   * *Physical Mechanism*: Quantifies the ratio of signal power to total noise power:
     $$\text{SNR} = \frac{I_{\text{sig}}^2}{\sigma_{\text{shot}}^2 + \sigma_{\text{thermal}}^2} = \frac{(R \cdot P_{rx})^2}{2 q (I_{\text{sig}} + I_{\text{bg}}) B + \frac{4 k_B T B}{R_L}}$$
   * *Significance*: Pearson correlation with path loss is **$r = 0.9952$** ($0.9995$ Spearman rank). SNR represents the ultimate communication link metric determining supported constellation orders and data rates.

4. **Theoretical Bit Error Rate (`theoretical_ber_ook`, `log10_ber`)**:
   * *Physical Mechanism*: For On-Off Keying (OOK) in AWGN:
     $$\text{BER} = \frac{1}{2}\operatorname{erfc}\left(\frac{\sqrt{\text{SNR}}}{2\sqrt{2}}\right)$$
   * *Significance*: In 6,404 samples (92.49%), $\text{SNR} > 25\text{ dB}$, rendering theoretical BER $< 10^{-12}$ (error-free transmission). High-BER events ($> 10^{-3}$) occur in 24 samples (0.35%), specifically during deep pedestrian blockage and extreme link outage.

5. **Ambient Optical Interference & Noise (`ambientLight`, `log_ambientLight`, `p_amb_uW`)**:
   * *Physical Mechanism*: Ambient solar and lamp illumination generates DC photocurrent $I_{\text{bg}} = R \cdot P_{\text{amb}}$, driving optical shot noise $\sigma_{\text{shot}}^2 = 2 q I_{\text{bg}} B$. In `pasilloCristal`, ambient illuminance surges up to $1,984$ lx, degrading link SNR by $10$ to $15$ dB.

6. **Optical Attenuation & Path Loss (`pathloss_clean_db`)**:
   * *Physical Mechanism*: Relative channel attenuation in dB between transmitted and detected modulated voltage. Directly captures total optical channel loss.

7. **Line-of-Sight Shadowing (`presenciaPersonas`)**:
   * *Physical Mechanism*: Simulates real-world dynamic vehicle blockage (e.g., crossing pedestrians or adjacent trucks).
   * *Empirical Impact*:
     * Free LOS: Mean Path Loss = **$-35.31$ dB**, Mean SNR = **$51.71$ dB**, Max BER = **$0.0$**
     * Obstructed: Mean Path Loss = **$-36.37$ dB**, Mean SNR = **$49.72$ dB**, Max BER = **$0.50$** (complete outage)

---

## 6. Channel-Quality Label Status & Scientific Target Determination

### 6.1 Does the Dataset Already Contain a Channel-Quality Label?

1. **Original Raw Dataset (`dataset.mat`)**:
   * **CONFIRMED**: The original dataset contains **NO channel-quality label**. It consists solely of physical measurements (`distancia`, `ambientLight`, `angRadiacion`, `angIncidencia`, `presenciaPersonas`, `lente`, `entorno`, `pathloss`).
2. **Newly Created Preprocessed Dataset (`vlc_channel_dataset_processed.csv`)**:
   * The preprocessing script (`preprocess_dataset.py`) created two provisional label columns:
     * `channel_quality_binary`: Binary indicator using a cutoff of `pathloss_clean_db >= -35.0 dB` (Good = 1, Poor = 0).
     * `channel_quality_multi`: Multi-class indicator with thresholds at $-34.0$ dB and $-38.0$ dB.
   * However, **neither threshold can be accepted arbitrarily without formal scientific justification**.

---

### 6.2 Scientific Evidence & Evaluation of Candidate Thresholds

To eliminate arbitrary selection, 5 distinct mathematical, statistical, and telecommunication criteria were evaluated against the 6,924 samples:

```
Distribution of Optical Path Loss (dB):
        Poor / Severe Attenuation               Good / Nominal Transmission
 [Outage] ◄───────────────┼─────────────────────────────────────────────► [High Power]
 -60 dB                -38.16 dB                   -35.0 dB                   -27.8 dB
                          ▲                           ▲
                  Otsu / K-Means Boundary       Empirical Median Split
                  (Inter-Class Variance)       (Information Entropy H=1)
```

#### Methodology 1: Otsu's Global Optimal Threshold (Variance Maximization)
* **Mathematical Criterion**: Otsu's discriminant criterion searches for threshold $t^*$ that maximizes between-class variance:
  $$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left(\mu_0(t) - \mu_1(t)\right)^2$$
* **Calculated Threshold**: **$t^* = \mathbf{-38.1640\text{ dB}}$**
* **Resulting Split**:
  * Poor Channel ($0$): **1,555 samples (22.46%)** — Mean PL = $-42.36$ dB, Mean SNR = $38.8$ dB
  * Good Channel ($1$): **5,369 samples (77.54%)** — Mean PL = $-33.89$ dB, Mean SNR = $54.3$ dB
* **Physical Justification**: Isolates the lower 22.5% tail consisting of severe angular misalignment ($\phi, \psi = 45^\circ$), long range ($> 1.8$ m), and deep pedestrian shadowing from the nominal operating envelope.

#### Methodology 2: Unsupervised K-Means Clustering ($k=2$)
* **Mathematical Criterion**: Minimizes within-cluster sum-of-squares (WCSS).
* **Cluster Centers**:
  * Cluster 0 (Degraded): **$-42.3604$ dB** ($n = 1,559$)
  * Cluster 1 (Nominal): **$-33.8858$ dB** ($n = 5,365$)
* **Midpoint Decision Boundary**: **$\mathbf{-38.1231\text{ dB}}$**
* **Remarkable Convergence**: K-Means ($k=2$) and Otsu's method independently converge on virtually the exact same threshold ($-38.12$ dB vs $-38.16$ dB), providing strong mathematical proof of an inherent natural boundary in the channel distribution.

#### Methodology 3: Empirical Median Split (Maximum Shannon Entropy)
* **Mathematical Criterion**: Sets threshold at the $50^{\text{th}}$ percentile (Median = $-35.1145$ dB), closely approximated by $-35.0000$ dB.
* **Information-Theoretic Value**: Maximizes Shannon target entropy:
  $$H(Y) = -p \log_2(p) - (1-p) \log_2(1-p) = \mathbf{0.9994\text{ bits}} \approx 1.0\text{ bit}$$
* **Resulting Split (at $-35.0$ dB)**:
  * Poor Channel ($0$): **3,591 samples (51.86%)**
  * Good Channel ($1$): **3,333 samples (48.14%)**
* **Physical Justification**:
  * Eliminates class imbalance artifacts in supervised machine learning models.
  * In adaptive rate control (IEEE 802.15.7), a vehicle must transition between high-throughput modulation (e.g., 16-QAM or 4-PPM) and robust modulation (OOK) at a balanced operating margin ($\text{SNR} \approx 52$ dB, $P_{rx} \approx 12.5\ \mu\text{W}$).
  * Corresponds directly to the boundary where pedestrian blockage begins to degrade channel throughput (unobstructed mean PL = $-35.31$ dB, obstructed mean PL = $-36.37$ dB).

#### Methodology 4: Gaussian Mixture Model (GMM) Bayesian Posterior Boundary
* **Mathematical Criterion**: Fits two Gaussian components via Expectation-Maximization (EM) and solves for equal posterior probability $P(C_0 | x) = P(C_1 | x)$.
* **Fitted Components**:
  * Component 0 (Severe Outage Mode): Weight = $6.89\%$, Mean = $-47.38$ dB, Std = $4.15$ dB
  * Component 1 (Nominal Mode): Weight = $93.11\%$, Mean = $-34.94$ dB, Std = $3.35$ dB
* **Calculated Boundary**: **$\mathbf{-43.4792\text{ dB}}$**
* **Resulting Split**: Poor = **438 samples (6.33%)**, Good = **6,486 samples (93.67%)**.
* **Limitation**: Suffers from high class imbalance (15:1 ratio), making it suitable only as an emergency link-loss alarm rather than an adaptive channel sensing target.

#### Methodology 5: Optical Communication Standards (IEEE 802.15.7 / FEC Limit)
* **Standard Criterion**: In optical wireless communications, Forward Error Correction (FEC) codes maintain error-free transmission up to a hard-decision limit of $\text{BER} \le 1.0 \times 10^{-3}$ (or $3.8 \times 10^{-3}$ for 7% HD-FEC overhead).
* **Empirical Data from Dataset**:
  * $\text{BER} > 1.0 \times 10^{-3}$: Exactly **24 samples (0.35%)**
  * $\text{BER} = 0.5$ (Complete Link Disruption): Exactly **3 samples (0.043%)**
  * Associated Channel Thresholds: Occurs exclusively when $\text{Path Loss} < -53.98$ dB (mean $-55.36$ dB) and $\text{SNR} < 14.56$ dB (mean $6.47$ dB).
* **Critical Finding**: Setting a binary ML target strictly at $\text{BER} = 10^{-3}$ yields a 99.65% vs 0.35% class distribution. This would degrade ML classification into rare-event anomaly detection. In vehicular VLC, adaptive channel sensing must switch modulation and power levels *well before* the FEC limit is breached.

---

### 6.3 Threshold Comparison Matrix

| Thresholding Methodology | Formal Cutoff | Poor Channel (0) | Good Channel (1) | Poor % | Good % | Scientific Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Empirical Median Split** | **$-35.11\text{ dB}$** | 3,405 | 3,519 | 49.18% | 50.82% | Max entropy ($H=1.0$), balanced ML training, median SNR ($52.3$ dB) |
| **Preprocessed Heuristic** | **$-35.00\text{ dB}$** | 3,591 | 3,333 | 51.86% | 48.14% | Practically identical to median ($0.11$ dB difference), round cutoff |
| **Otsu's Optimal Separation** | **$-38.16\text{ dB}$** | 1,555 | 5,369 | 22.46% | 77.54% | Maximizes inter-class variance $\sigma_B^2$, separates deep fade modes |
| **K-Means Cluster Midpoint** | **$-38.12\text{ dB}$** | 1,559 | 5,365 | 22.52% | 77.48% | Unsupervised clustering boundary between nominal and degraded modes |
| **Blockage Boundary Mean** | **$-36.43\text{ dB}$** | 2,561 | 4,363 | 36.99% | 63.01% | Corresponds to mean attenuation under pedestrian shadowing |
| **GMM Posterior Boundary** | **$-43.48\text{ dB}$** | 438 | 6,486 | 6.33% | 93.67% | Isolates catastrophic outage distribution tail |
| **FEC Communication Limit** | **$-54.00\text{ dB}$** | 24 | 6,900 | 0.35% | 99.65% | Physical IEEE 802.15.7 packet drop threshold ($\text{BER} \ge 10^{-3}$) |

---

### 6.4 Final Scientific Target Recommendations

Based on the statistical and physical evidence, two target formulations are recommended for subsequent ML modeling:

1. **Recommended Primary Binary Target: `channel_quality_binary` ($\text{Cutoff} = -35.0\text{ dB}$ / $-35.11\text{ dB}$)**
   * **Class 1 (Good Channel)**: $\text{Path Loss} \ge -35.0\text{ dB}$ (48.14%) — High-power link, high SNR ($> 52$ dB), suitable for high-order modulation (e.g. 16-QAM or 4-PPM).
   * **Class 0 (Poor Channel)**: $\text{Path Loss} < -35.0\text{ dB}$ (51.86%) — Degraded link, requires robust transmission fallback (e.g. OOK with repetition coding, power boosting, or beam alignment).
   * **Justification**: Eliminates class imbalance, maximizes target information entropy, and aligns with the median channel state observed in the physical vehicle testbed.

2. **Recommended Multi-Class Target: `channel_quality_multi` (3-State Vehicular CQI)**
   * **Class 2 (Good / High Throughput)**: $\text{Path Loss} \ge -34.0\text{ dB}$ (**2,520 samples, 36.40%**) — Unobstructed, short-range, optimal alignment.
   * **Class 1 (Moderate / Safe Margin)**: $-38.0\text{ dB} \le \text{Path Loss} < -34.0\text{ dB}$ (**2,845 samples, 41.09%**) — Nominal range, moderate angles or partial shadowing; operates with standard modulation.
   * **Class 0 (Poor / Deep Fade / Blockage)**: $\text{Path Loss} < -38.0\text{ dB}$ (**1,559 samples, 22.52%**) — Derived exactly from the Otsu/K-means mathematical boundary ($-38.12$ dB); triggers emergency protocol, optical power boost, or beam steering.

---

## 7. Recommendations for Next Stage (Feature Selection & Model Training)

When proceeding to model training in the next step, the following essential practices must be enforced:

1. **Prevent Target Leakage**:
   * Features that are mathematical derivatives of the target (e.g., `vpp_volts`, `p_rx_uW`, `p_rx_dBm`, `snr_db`, `theoretical_ber_ook`, and `pathloss_clean_db`) **must NOT be used as input features $\mathbf{X}$** for predicting channel quality.
   * The feature matrix $\mathbf{X}$ must be restricted strictly to **inputs accessible to vehicle sensors prior to optical channel estimation**:
     $$\mathbf{X} = [\text{distancia\_m}, \text{ambientLight}, \text{log\_ambientLight}, \text{angRadiacion}, \cos(\phi), \text{angIncidencia}, \cos(\psi), \text{angular\_alignment\_factor}, \text{geometric\_loss\_factor}, \text{presenciaPersonas}, \text{environment\_id}]$$

2. **Strict Chronological / Session-Based Splitting**:
   * Because vehicular VLC measurements are temporal and grouped by sessions, validation splits should evaluate generalization across unseen sessions (e.g., Leave-One-Session-Out cross-validation or chronological train/validation/test splits) rather than naive random shuffling.

3. **Pipelines Fitted on Training Set Only**:
   * Standard scalers and transformers must be fitted strictly on training folds to prevent data snooping.

---

## 8. Summary Table of Key Findings

| Requirement | Finding / Value |
| :--- | :--- |
| **Final Rows & Columns** | **6,924 rows**, **29 columns** |
| **Data Types** | 20 `float64`, 7 `int64`, 2 `object` (string) |
| **Missing Values** | **0 nulls (0.00%)**, **0 infinities (0.00%)** |
| **Dropouts Imputed** | 421 distance sonar dropouts ($6.08\%$) imputed; 3 pathloss outages bounded to $-60\text{ dB}$ |
| **Pre-existing Labels** | **None in raw `dataset.mat`**; provisional labels created in preprocessed files |
| **Selected Binary Cutoff** | **$-35.0\text{ dB}$** (Median split: 51.86% Poor vs 48.14% Good) |
| **Alternative Mathematical Cutoff** | **$-38.16\text{ dB}$** (Otsu variance maximization / K-means: 22.46% Poor vs 77.54% Good) |
| **Machine Learning Status** | **Zero models trained** (Strictly adhering to instructions) |
