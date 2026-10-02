# Machine Learning Model Training & Evaluation Report
## AI-Based Adaptive Car-to-Car Visible Light Communication with Channel Sensing

---

### Executive Summary

This report documents the machine learning model training, cross-validation, and held-out benchmarking phase for binary optical channel quality classification in vehicular Visible Light Communication (VLC). 

The goal of this pipeline is to sense physical environmental and geometric variables **prior to optical reception** and accurately predict whether the optical channel is **Good** ($\text{Pathloss} \ge -35.0\text{ dB}$) or **Poor** ($\text{Pathloss} < -35.0\text{ dB}$). This prediction dynamically dictates the physical-layer modulation format:
- **Good Channel** $\longrightarrow$ **On-Off Keying (OOK)**: Maximum throughput, minimal processing delay, bandwidth-efficient.
- **Poor Channel** $\longrightarrow$ **Binary Shift Keying with Forward Error Correction (BSK + FEC)**: Robust subcarrier modulation providing $3\text{--}6\text{ dB}$ coding gain against severe attenuation, pedestrian shadow fading, and solar shot noise.

The pipeline was executed with strict adherence to scientific zero-leakage standards, group-based session partitioning to prevent trajectory autocorrelation leakage, and comprehensive multi-metric evaluation across six candidate models.

---

### 1. Dataset Specification

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Source File** | `dataset/ml_ready/vlc_channel_ml_ready.csv` | Formatted ML-ready feature matrix |
| **Total Samples** | $6,924$ rows | Validated experimental time-series observations |
| **Total Features** | $6$ physical input features | Measurable independently of optical signal reception |
| **Missing Values** | $0$ ($0.0\%$) | Complete dataset, zero imputation required |
| **Target Variable** | `target_channel_quality_binary` | Binary class: $1 = \text{Good Channel}$, $0 = \text{Poor Channel}$ |
| **Experimental Sessions** | $12$ distinct sessions | Continuous trajectories in lab and glass corridor |
| **Physical Environments** | $2$ environments | `laboratorio` ($N=6,419$) and `pasilloCristal` ($N=505$) |

---

### 2. Predictor Features & Target Definition

#### 2.1 Predictor Feature Set (Vehicle-Available Sensory Inputs)

All input features represent physical conditions available to the vehicle via onboard sensors (ultrasonic/radar distance, ambient photodetector/luxmeter, angular positioning, and camera/LIDAR pedestrian detection) **before or independently of optical signal decoding**:

1. `distancia_m` (Continuous, float, meters): Distance between optical transmitter and optical receiver. Range: $[0.01\text{ m}, 2.37\text{ m}]$.
2. `ambientLight` (Continuous, float, lux): Background ambient optical illuminance (fluorescent lights, outdoor daylight). Range: $[2.0\text{ lx}, 1,984.0\text{ lx}]$.
3. `angRadiacion` (Continuous, float, degrees): Optical transmitter half-angle / emission radiation angle. Range: $[0.0^\circ, 15.0^\circ]$.
4. `angIncidencia` (Continuous, float, degrees): Optical receiver incident angle relative to detector normal. Range: $[0.0^\circ, 20.0^\circ]$.
5. `presenciaPersonas` (Discrete binary, integer): Human/pedestrian obstruction state ($0 = \text{Clear line-of-sight}$, $1 = \text{Pedestrian blockage}$).
6. `entorno_encoded` (Discrete binary, integer): Physical propagation environment ($0 = \text{laboratorio}$, $1 = \text{pasilloCristal}$).

#### 2.2 Strict Zero Target Leakage Verification

Under no circumstances were received optical signal measurements used as predictors. The following variables were audited and strictly excluded from the feature matrix:
- `pathloss` / `pathloss_clean_db` ($\text{dB}$)
- `vpp_volts` (Peak-to-peak received electrical signal voltage)
- `p_rx_uW` (Received optical power)
- `snr_db` (Signal-to-Noise Ratio)
- `BER` (Bit Error Rate)
- `outage` / `margin_db`

**Audit Result:** Zero leakage confirmed. The predictor matrix contains exclusively independent physical sensory features.

#### 2.3 Target Ground-Truth Construction

The binary channel quality target was established based on empirical optical channel pathloss:
$$\text{Class} = \begin{cases} 1 \quad (\text{Good Channel}), & \text{if } \text{pathloss\_clean\_db} \ge -35.0\text{ dB} \\ 0 \quad (\text{Poor Channel}), & \text{if } \text{pathloss\_clean\_db} < -35.0\text{ dB} \end{cases}$$

- **Poor Channel ($0$):** $3,591$ samples ($51.86\%$)
- **Good Channel ($1$):** $3,333$ samples ($48.14\%$)
- **Class Balance:** Approximately $52:48$, representing an inherently balanced classification problem with zero artificial skew.

---

### 3. Train/Test Splitting & Validation Methodology

#### 3.1 Session-Grouped Partitioning (Leakage Prevention)

The dataset consists of continuous vehicular movement recordings grouped across 12 distinct experimental sessions. In continuous sensor telemetry, adjacent samples ($t$ and $t+\Delta t$) exhibit strong temporal and spatial auto-correlation. 

A standard random stratified split randomly scatters neighboring samples across train and test partitions, causing artificial data leakage and overly optimistic test scores ($>96\%$ accuracy). To measure genuine generalization to unseen vehicular trajectories and conditions, **`GroupShuffleSplit`** on `session_id` was employed:

- **Held-Out Test Partition ($24.5\%$ of dataset, $N = 1,697$):**
  - Sessions: `[5, 6, 10]`
  - Poor Channel ($0$): $869$ samples ($51.21\%$)
  - Good Channel ($1$): $828$ samples ($48.79\%$)
  - Balanced distribution across both classes and both line-of-sight/obstructed conditions.
- **Training Partition ($75.5\%$ of dataset, $N = 5,227$):**
  - Sessions: `[1, 2, 3, 4, 7, 8, 9, 11, 12]`
  - Poor Channel ($0$): $2,722$ samples ($52.08\%$)
  - Good Channel ($1$): $2,505$ samples ($47.92\%$)
  - Contains both environments (`laboratorio` and `pasilloCristal` in Session 12), full distance range ($0.01\text{--}2.37\text{ m}$), and full ambient illuminance range ($2\text{--}1,984\text{ lx}$).

#### 3.2 Preprocessing Pipeline

To eliminate data leakage during feature transformation, all scalers were embedded inside a scikit-learn `Pipeline` and fitted solely on the training partition:
- Continuous features (`distancia_m`, `ambientLight`, `angRadiacion`, `angIncidencia`) $\longrightarrow$ `StandardScaler()`.
- Discrete binary features (`presenciaPersonas`, `entorno_encoded`) $\longrightarrow$ Passthrough ($0/1$).

---

### 4. Candidate Models Benchmarked

The following supervised learning architectures were trained and evaluated:
1. **Logistic Regression:** Regularized linear model ($L_2$ penalty, $C=1.0$, `max_iter=1000`). Serves as a transparent linear baseline.
2. **Decision Tree Classifier:** CART decision tree (`max_depth=6`, `min_samples_split=10`, `random_state=42`). Provides human-interpretable orthogonal decision rules.
3. **Random Forest Classifier:** Bagged ensemble of $150$ decorrelated decision trees (`n_estimators=150`, `max_depth=8`, `min_samples_split=5`, `random_state=42`). Mitigates tree variance and provides robust non-linear boundaries.
4. **Gradient Boosting Classifier:** Sequential ensemble boosting residuals via deviance loss (`n_estimators=150`, `max_depth=4`, `learning_rate=0.1`, `random_state=42`).
5. **XGBoost Classifier:** Extreme Gradient Boosting with exact greedy tree pruning and $L_2$ regularization (`n_estimators=150`, `max_depth=4`, `learning_rate=0.1`, `eval_metric='logloss'`).
6. **Naive Baseline:** Zero-rule dummy classifier (`strategy='most_frequent'`) to provide an empirical floor.

---

### 5. Benchmark Results & Comparative Analysis

All models were evaluated on the held-out test sessions ($N = 1,697$). Cross-validation accuracy represents 5-fold stratified CV on the training partition ($N = 5,227$).

| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Precision (Good) | Recall (Good) | F1-Score (Good) | ROC-AUC | True Neg (Poor) | False Pos | False Neg | True Pos (Good) | Fit Time | Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | $94.89\% \pm 0.73\%$ | **82.56%** | **89.00%** | $73.31\%$ | **80.40%** | **0.9100** | **794** | **75** | $221$ | $607$ | $0.688\text{ s}$ | $51.3\ \mu\text{s}$ |
| **XGBoost** | $95.12\% \pm 0.79\%$ | **81.38%** | $80.70\%$ | $81.28\%$ | **80.99%** | **0.8918** | $708$ | $161$ | $155$ | $673$ | $0.300\text{ s}$ | $35.2\ \mu\text{s}$ |
| **Gradient Boosting** | $95.39\% \pm 0.64\%$ | **80.85%** | $79.55\%$ | $81.76\%$ | **80.64%** | **0.8790** | $695$ | $174$ | $151$ | $677$ | $0.887\text{ s}$ | $34.1\ \mu\text{s}$ |
| **Logistic Regression** | $80.28\% \pm 2.04\%$ | **74.43%** | $82.51\%$ | $60.39\%$ | **69.74%** | **0.7831** | $763$ | $106$ | $328$ | $500$ | $0.049\text{ s}$ | $2.8\ \mu\text{s}$ |
| **Decision Tree** | $90.34\% \pm 0.63\%$ | **72.60%** | $66.25\%$ | **89.37%** | **76.09%** | **0.8724** | $492$ | $377$ | **88** | **740** | $0.017\text{ s}$ | $2.4\ \mu\text{s}$ |
| **Naive Baseline** | $52.08\% \pm 0.02\%$ | **51.21%** | $0.00\%$ | $0.00\%$ | $0.00\%$ | **0.5000** | $869$ | $0$ | $828$ | $0$ | $0.009\text{ s}$ | $0.9\ \mu\text{s}$ |

*Note: All metric values are exact and derived from `results/model_comparison.csv`.*

---

### 6. Confusion Matrix & ROC-AUC Analysis

#### 6.1 Confusion Matrix Breakdown

The individual confusion matrices demonstrate how prediction errors are distributed between the two channel states:

- **Random Forest:**
  - True Negatives ($TN$): $794$ ($46.8\%$ of test set)
  - False Positives ($FP$): $75$ ($4.4\%$ of test set)
  - False Negatives ($FN$): $221$ ($13.0\%$ of test set)
  - True Positives ($TP$): $607$ ($35.8\%$ of test set)
  - **Key Observation:** Random Forest achieves the lowest False Positive count ($FP=75$) among all non-naive models.

- **Decision Tree:**
  - $TN = 492$, $FP = 377$, $FN = 88$, $TP = 740$.
  - Generates an excessive number of False Positives ($377$), misclassifying over $43\%$ of poor channels as good.

- **Gradient Boosting & XGBoost:**
  - Balanced sensitivity and specificity ($FP \approx 160\text{--}174$, $FN \approx 151\text{--}155$), maintaining strong F1 parity across both classes.

#### 6.2 Receiver Operating Characteristic (ROC)

The ROC curve benchmarks the True Positive Rate against the False Positive Rate across all probability decision thresholds:
- **Random Forest:** $\text{AUC} = \mathbf{0.9100}$ (Outstanding discriminative separation)
- **XGBoost:** $\text{AUC} = \mathbf{0.8918}$
- **Gradient Boosting:** $\text{AUC} = \mathbf{0.8790}$
- **Decision Tree:** $\text{AUC} = \mathbf{0.8724}$
- **Logistic Regression:** $\text{AUC} = \mathbf{0.7831}$
- **Naive Baseline:** $\text{AUC} = \mathbf{0.5000}$

Random Forest dominates the high-specificity region ($\text{FPR} < 0.15$), confirming that it maintains high confidence when identifying clear channels.

---

### 7. Feature Importance Analysis

Feature importances were extracted across all tree-based architectures (`Random Forest`, `Gradient Boosting`, `XGBoost`, `Decision Tree`):

| Physical Predictor | Random Forest Weight | Gradient Boosting Weight | XGBoost Weight | Physical Communication Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| `distancia_m` | **58.4%** | **74.8%** | **47.6%** | Free-space optical pathloss follows $1/d^2$ inverse-square power roll-off. Dominates DC channel gain $H(0)$. |
| `angIncidencia` | **18.7%** | **11.2%** | **23.1%** | Receiver effective optical collection area scales with $\cos(\psi)$. Steep angles cause severe optical clipping. |
| `angRadiacion` | **11.8%** | **7.6%** | **14.8%** | Transmitter Lambertian emission pattern scales with $\cos^m(\phi)$. Off-axis misalignment degrades signal. |
| `ambientLight` | **6.5%** | **3.8%** | **7.5%** | Solar and fluorescent illumination creates optical shot noise, degrading SNR and raising the effective noise floor. |
| `presenciaPersonas` | **4.6%** | **2.6%** | **7.0%** | Human body blockage introduces localized shadow fading ($10\text{--}20\text{ dB}$ optical attenuation). |
| `entorno_encoded` | **0.0%** | **0.0%** | **0.0%** | Constant in test sessions ($[5, 6, 10]$); serves as macro-environment flag for reflective sidewalls. |

**Key Physical Finding:**
Propagation distance (`distancia_m`) and geometric orientation angles (`angIncidencia`, `angRadiacion`) account for over **$85\%$** of the decision weight, directly reflecting the optical channel physics governed by the Lambertian channel equation:
$$H(0) = \frac{(m+1) A_{\text{det}}}{2\pi d^2} \cos^m(\phi) \cos(\psi) T_s(\psi) g(\psi)$$

---

### 8. Champion Model Selection Rationale

The selected champion model is **Random Forest Classifier** (`models/channel_quality_model.pkl`).

#### Why Random Forest Over Pure Accuracy or Gradient Boosting?

In safety-critical vehicular VLC communication, symmetric classification error is an invalid assumption. The operational cost of prediction errors is asymmetric:

1. **The Cost of a False Positive (Predicting "Good" when actual channel is "Poor"):**
   - The vehicle transmitter selects **OOK** modulation without FEC coding.
   - Because the actual channel is attenuated or obstructed, received optical power drops below the photodetector threshold.
   - The resulting Bit Error Rate exceeds the maximum limit ($\text{BER} > 10^{-2}$), causing complete packet drops, communication blackouts, or corrupted platooning telemetry.
   - **Operational Impact:** **Catastrophic packet loss and safety hazard.**

2. **The Cost of a False Negative (Predicting "Poor" when actual channel is "Good"):**
   - The vehicle transmitter selects **BSK + FEC** modulation.
   - Because the channel is actually clear, transmission succeeds with zero errors, albeit with slightly reduced data rate.
   - **Operational Impact:** **Benign throughput penalty with zero packet loss.**

#### Quantitative Justification:
- **Lowest False Positives:** Random Forest produced only **$75$ False Positives** out of $869$ poor channels (an error rate of only $8.6\%$), compared to $161$ for XGBoost ($+114\%$) and $174$ for Gradient Boosting ($+132\%$).
- **Highest Good-Channel Precision:** **$89.00\%$**, ensuring that when the transmitter is commanded to use high-throughput OOK, the channel is genuinely clear $89$ times out of $100$.
- **Highest Overall ROC-AUC:** **$0.9100$**, proving superior discrimination across all operational thresholds.
- **Real-Time Latency:** Average inference time of **$51.3\ \mu\text{s}$ per sample** ($< 0.1\text{ ms}$), easily exceeding the sub-millisecond execution requirement for automotive Electronic Control Units (ECUs).

---

### 9. System & Physical Limitations

1. **Indoor Experimental Grounding:** The underlying dataset was collected using laboratory and glass corridor test tracks. Outdoor phenomena such as atmospheric turbulence, thick fog, heavy rain, and solar background noise exceeding $10,000\text{ lx}$ will require supplementary calibration.
2. **Environment Cardinality:** Only one session was recorded in `pasilloCristal`. While the physical variables (distance, angles, lux) generalized effectively, outdoor urban canyons with diverse wall reflectivities require additional multi-environment training data.
3. **Boundary Transition Zone:** Samples near the $-35.0\text{ dB}$ decision threshold ($\pm 1.0\text{ dB}$) experience marginal classification confidence. In production, a hysteresis margin or probabilistic threshold ($P(\text{Good}) > 0.65$) should be enforced before transitioning to OOK.
4. **Dynamic Angular Misalignment:** High-frequency vehicle chassis vibrations and pitch/roll variations during road bumps introduce transient angular excursions that require high-rate inertial sensor filtering.

---

### 10. Generated Artifacts & File Structure

```
DVLC_VLC_ML_Animation/
├── dataset/
│   └── ml_ready/
│       └── vlc_channel_ml_ready.csv         # Verified ML-ready dataset (6,924 rows)
├── models/
│   └── channel_quality_model.pkl            # Champion Random Forest pipeline (joblib)
├── results/
│   ├── model_comparison.csv                 # Metrics table for all benchmarked models
│   ├── roc_curve.png                        # Comparative ROC curves (AUC = 0.9100)
│   ├── feature_importance.png               # Physical predictor weight distributions
│   ├── classification_report.txt            # Full precision/recall/F1 breakdowns
│   └── confusion_matrices/
│       ├── random_forest_cm.png             # Champion confusion matrix (FP = 75)
│       ├── xgboost_cm.png                   # XGBoost confusion matrix
│       ├── gradient_boosting_cm.png         # Gradient Boosting confusion matrix
│       ├── decision_tree_cm.png             # Decision Tree confusion matrix
│       ├── logistic_regression_cm.png       # Logistic Regression confusion matrix
│       ├── naive_baseline_cm.png            # Naive Baseline confusion matrix
│       └── confusion_matrices_all.png       # Combined 2x3 benchmark grid
├── scripts/
│   ├── train_final_models.py                # Reproducible training & evaluation script
│   └── predict_channel.py                   # Standalone inference & modulation selector
└── reports/
    └── ML_TRAINING_REPORT.md                # This complete scientific report
```

---

### Conclusion & Operational Readiness

The ML classification pipeline has been verified. The champion **Random Forest** model achieves **$82.56\%$ out-of-session test accuracy**, **$89.00\%$ precision**, and **$0.9100$ ROC-AUC** while maintaining zero target leakage. The standalone inference module `scripts/predict_channel.py` is operational and ready to serve real-time predictions for downstream vehicular VLC link adaptation.
