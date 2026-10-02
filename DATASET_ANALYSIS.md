# Visible Light Communication (VLC) Dataset Analysis Report
## Academic Project: AI-Based Adaptive Car-to-Car Visible Light Communication with ML-Based Channel Sensing

---

### Executive Summary

This document provides a comprehensive structural, physical, and statistical analysis of the experimental dataset (`dataset.mat`) used for the project **AI-Based Adaptive Car-to-Car Visible Light Communication with ML-Based Channel Sensing**.

The dataset originates from an experimental Visible Light Communication (VLC) measurement platform built on an **Arduino Uno** architecture equipped with a **Thorlabs PDA100A2** switchable gain amplified photodetector, an **HC-SR04** ultrasonic distance sensor, and a **BH1750** digital ambient illuminance (lux) meter. The raw serial telemetry was acquired across 12 distinct experimental sessions on **May 11, 2022**, and subsequently processed in MATLAB via `genDataSet.m` using 10-sample downsampling windows to yield the unified table `tablaTotal` stored in `dataset.mat`.

---

## 1. Dataset Overview & Key Metrics

* **Source File**: `dataset/VLC-measurement-platform-main/VLC-measurement-platform-main/dataset.mat`
* **File Format**: MATLAB 5.0 MAT-file (Platform: MACI64, Created: Tue May 17 23:57:09 2022)
* **File Size**: 59,665 bytes (~58.27 KB)
* **Internal Structure**: MATLAB Object System (MCOS) serialized table (`tablaTotal`)
* **Total Sample Count**: **6,924 samples**
* **Total Features/Variables**: **9 columns**
* **Time Span**: May 11, 2022, 10:19:29.700 UTC to 11:47:14.290 UTC (~1 hr 28 min)

---

## 2. Feature Inventory & Technical Specification

| Variable Name | Physical Meaning | Acquisition Source / Sensor | Unit | Storage Type in MATLAB | Shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `date` | Timestamp of measurement window | Arduino / PC Serial Logger | `yyyy-MM-dd HH:mm:ss.SSS` | `datetime` (epoch ms) | (6924, 1) |
| `distancia` | Line-of-sight distance between transmitter & receiver | HC-SR04 Ultrasonic Sonar | Centimeters (cm) | `float64` | (6924, 1) |
| `angRadiacion` | Transmitter LED radiation / irradiance angle ($\phi$) | Pre-set mechanical stage | Degrees (°) | `float64` | (6924, 1) |
| `angIncidencia` | Receiver photodiode incidence angle ($\psi$) | Pre-set mechanical stage | Degrees (°) | `float64` | (6924, 1) |
| `ambientLight` | Background ambient optical illumination | BH1750 Digital Lux Sensor | Lux (lx) | `float64` | (6924, 1) |
| `presenciaPersonas` | Presence of pedestrians / line-of-sight blockage | Experimental control log | Binary ($0$ or $1$) | `float64` | (6924, 1) |
| `lente` | Optical lens / concentrator configuration | Experimental control log | Flag ($0$ = no lens) | `float64` | (6924, 1) |
| `entorno` | Physical experimental environment | Test location classification | Categorical string | `category` (uint8 code) | (6924, 1) |
| `pathloss` | VLC optical channel attenuation / path loss | Calculated from PDA100A2 $V_{pp}$ | Decibels (dB) | `float64` | (6924, 1) |

---

## 3. Comprehensive Summary Statistics

The table below details the descriptive statistics computed directly from the 6,924 rows of `dataset.mat`:

| Variable Name | Count | Missing | Zeros | Min | 25% | Median (50%) | Mean | 75% | Max | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `distancia` (cm) | 6,924 | 0 | 421 | 0.000 | 85.800 | 123.200 | 116.401 | 157.000 | 236.600 | 50.892 |
| `angRadiacion` (°) | 6,924 | 0 | 2,416 | 0.000 | 0.000 | 30.000 | 24.127 | 45.000 | 45.000 | 18.669 |
| `angIncidencia` (°) | 6,924 | 0 | 4,592 | 0.000 | 0.000 | 0.000 | 12.312 | 30.000 | 45.000 | 17.809 |
| `ambientLight` (lx) | 6,924 | 0 | 0 | 1.919 | 23.330 | 40.000 | 115.973 | 49.167 | 1,984.082 | 298.879 |
| `presenciaPersonas` | 6,924 | 0 | 3,770 | 0.000 | 0.000 | 0.000 | 0.456 | 1.000 | 1.000 | 0.498 |
| `lente` | 6,924 | 0 | 6,924 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pathloss` (dB) [finite] | 6,921 | 0 | 0 | -56.990 | -37.447 | -35.114 | -35.783 | -32.807 | -27.841 | 4.615 |

> **Note on `pathloss`**: 3 samples out of 6,924 evaluate to `-inf` because $V_{pp} = 0.0$ V (complete signal loss / zero modulation), producing $\log_{10}(0) = -\infty$.

---

## 4. In-Depth Channel Measurement Analysis

### 4.1 Distance Measurements (`distancia`)
* **Sensor**: HC-SR04 Ultrasonic Sonar Module (operational beam angle ~15°, nominal range 2 cm – 400 cm; clamped to 350 cm in `measurment_platform.ino`).
* **Non-Zero Distribution**:
  * Samples with $d > 0$: **6,503 samples (93.92%)**
  * Minimum non-zero distance: **0.70 cm**
  * Maximum distance: **236.60 cm (~2.37 m)**
  * Non-zero Mean: **123.94 cm**, Median: **127.90 cm**, Std: **42.70 cm**
* **Zero Values ($0.0$ cm)**:
  * Exactly **421 samples (6.08%)** have a distance of $0.0$ cm.
  * *Root Cause*: The HC-SR04 returns 0 when no ultrasonic echo is detected within the timeout window. This occurs during angular misalignment ($\phi, \psi \ge 30^\circ$) or when the inter-vehicle test rig exceeded the ultrasonic reflection cone.
  * In ML preprocessing, these zero values must be imputed (or filtered) rather than treated as physical 0 cm contact.

### 4.2 Ambient Light Measurements (`ambientLight`)
* **Sensor**: BH1750 Digital Ambient Light Sensor (calibrated illuminance in lux, I2C interface, range 1 – 65,535 lx).
* **Observed Range**: **1.919 lx to 1,984.082 lx**
* **Distribution Profile**:
  * 75% of measurements are below **49.17 lx** (indoor enclosed laboratory ambient lighting).
  * High-illuminance tail (up to 1,984.08 lx) occurs exclusively during session 12 in `pasilloCristal` (glass-walled corridor exposed to outdoor sunlight).
* **Physical Significance in VLC**: Ambient light is the dominant source of optical background shot noise in vehicle VLC:
  $$\sigma_{\text{shot}}^2 = 2 q R P_{\text{amb}} B$$
  High ambient light (e.g. sunlight or high lux corridor) degrades the signal-to-interference-plus-noise ratio (SINR) and can drive the receiver photodiode into saturation.

### 4.3 Transmitter Radiation Angle (`angRadiacion`)
* **Physical Parameter**: Irradiance angle $\phi$ measured from the transmitter normal axis to the receiver line-of-sight vector.
* **Observed Values**: Discrete set $\{0^\circ, 30^\circ, 45^\circ\}$
  * **$0^\circ$**: 2,416 samples (34.89%)
  * **$30^\circ$**: 2,387 samples (34.47%)
  * **$45^\circ$**: 2,121 samples (30.63%)
* **Physical Significance**: Governs Lambertian radiant intensity:
  $$R_0(\phi) = \frac{m + 1}{2\pi} \cos^m(\phi)$$
  where $m = -\frac{\ln(2)}{\ln(\cos(\Phi_{1/2}))}$ is the Lambertian emission order. Larger angles exponentially reduce the received power.

### 4.4 Receiver Incidence Angle (`angIncidencia`)
* **Physical Parameter**: Angle $\psi$ between the incident optical ray and the photodetector surface normal.
* **Observed Values**: Discrete set $\{0^\circ, 30^\circ, 45^\circ\}$
  * **$0^\circ$**: 4,592 samples (66.32%)
  * **$30^\circ$**: 1,313 samples (18.96%)
  * **$45^\circ$**: 1,019 samples (14.72%)
* **Physical Significance**: Controls the effective light collection area of the photodiode:
  $$A_{\text{eff}}(\psi) = A_{\text{det}} \cos(\psi) \quad \text{for } 0 \le \psi \le \Psi_{\text{FOV}}$$
  Beyond the Field-of-View (FOV) limit, $A_{\text{eff}} = 0$, terminating the optical link.

### 4.5 Pedestrian Presence / Shadowing (`presenciaPersonas`)
* **Values**: Binary flag ($0.0$ or $1.0$).
  * **$0$ (Free Channel / No Obstruction)**: 3,770 samples (54.45%)
  * **$1$ (Obstructed Channel / Shadowing)**: 3,154 samples (45.55%)
* **Physical Significance**: Directly simulates real-world Car-to-Car / Vehicle-to-Vehicle (V2V) VLC scenarios where pedestrians, cyclists, or obstacles cross the direct optical beam, causing rapid deep fading and shadowing.

### 4.6 Lens / Concentrator Indicator (`lente`)
* **Values**: Exactly **0.0 for all 6,924 samples** (variance = 0.0).
* **Physical Significance**: No optical focusing lens or Winston cone concentrator was tested during these sessions.
* **ML Implication**: Zero variance feature; **must be dropped** from machine learning models.

### 4.7 Operating Environment (`entorno`)
* **Values**: Categorical with two distinct environments:
  * **`laboratorio` (Enclosed Lab)**: **6,419 samples (92.71%)** — Low ambient illuminance, minimal natural light.
  * **`pasilloCristal` (Glass Corridor)**: **505 samples (7.29%)** — High ambient illuminance, high multipath reflection from glass windows.

### 4.8 Path Loss (`pathloss`)
* **Metric**: Optical channel attenuation in dB.
* **Finite Values Range**: **-56.9897 dB to -27.8410 dB**
* **Finite Mean**: **-35.7835 dB**, **Std**: **4.6152 dB**
* **Singularities**: Exactly 3 samples (indices 1435, 1436, 1440) exhibit `-inf` dB due to complete loss of modulated signal ($V_{pp} = 0$).

---

## 5. Raw Data Generation & Signal Processing Pipeline (`genDataSet.m`)

The data generation process proceeds through three distinct stages:

```
[Hardware Platform]
Thorlabs PDA100A2 Photodiode (val: ADC 0-1023)
HC-SR04 Ultrasonic Sonar (distancia: cm)
BH1750 Lux Sensor (lux: lx)
Manual Configuration (angRadiacion, angIncidencia, personas, entorno)
               │
               ▼ (Serial Port @ 9600 baud)
12 Raw CSV Tables: tabla1_11052022.csv to tabla12_11052022.csv
               │
               ▼ (genDataSet.m: MATLAB Processing)
Windowing (wind = 10 samples):
  - Vpp = max(signal_window) - min(signal_window)
  - pathloss = 10 * log10( 0.6 * Vpp / (Vpp_input * 10^K) )
  - distancia = mean(measuredDistance(window))
  - angRadiacion = mean(measuredAngRadiacion(window))
  - angIncidencia = mean(measuredAngIncidencia(window))
  - ambientLight = mean(measuredLuxes(window))
  - presenciaPersonas = mean(measuredPersonas(window))
               │
               ▼
Unified Table: tablaTotal (6,924 rows x 9 columns) -> dataset.mat
```

### MATLAB Formula Analysis:
In `genDataSet.m`:
* Transmitted Peak-to-Peak Voltage: $V_{pp\_input} = 3.0$ V
* Photodiode Responsivity & Gain scaling constant: $0.6$
* Window size: $wind = 10$ raw samples

---

## 6. Critical Findings & Data Quality Assessment

### 6.1 Status of GOOD/BAD Channel Quality Labels
> **VERIFIED**: There are **NO GOOD/BAD channel labels** in `dataset.mat`.
> The dataset contains strictly raw sensor measurements, environmental identifiers, and computed physical path loss values.
> In a later stage, binary or multi-class channel states (e.g., `GOOD` vs `BAD` or `EXCELLENT` / `DEGRADED` / `OUTAGE`) must be formulated based on physical channel thresholds (such as path loss cutoff dB, SNR, or BER).

### 6.2 Data Inconsistencies & Flaws

1. **Mathematical Discrepancy in `genDataSet.m`**:
   * In **Table 1** (lines 39):
     $$\text{pathloss} = 10 \log_{10}\left(\frac{0.6 \cdot V_{pp}}{3 \times 10^6}\right)$$
   * In **Tables 2 through 12** (lines 73, 106, 138, 170, 202, 234, 266, 298, 330, 362, 394):
     $$\text{pathloss} = 10 \log_{10}\left(\frac{0.6 \cdot V_{pp}}{3 \times 10^5}\right)$$
   * *Impact*: Table 1 has a theoretical 10 dB offset compared to Tables 2–12 due to the denominator divisor ($10^6$ vs $10^5$). When training ML models, this offset should either be calibrated or normalized.

2. **Negative Infinite (`-inf`) Path Loss**:
   * Exactly 3 rows have $V_{pp} = 0$, producing $\log_{10}(0) = -\infty$.
   * *Remedy*: Clip `-inf` to a realistic channel floor (e.g., $-70.0$ dB) or treat as link outage.

3. **Zero Ultrasonic Sonar Readings (`distancia == 0.0`)**:
   * 421 samples (6.08%) report $0.0$ cm due to ultrasonic echo timeout.
   * *Remedy*: Impute using physical optical path loss inverse mapping or preceding rolling window values.

4. **Zero-Variance Feature (`lente`)**:
   * All 6,924 values are $0.0$.
   * *Remedy*: Exclude from feature matrix $\mathbf{X}$.

---

## 7. Recommended Machine Learning Input Features for Channel Sensing

For an adaptive Car-to-Car VLC system, the receiver must estimate the Channel State Information (CSI) or predict path loss in real time to adaptively select modulation schemes (e.g., OOK, 4-PPM, 16-QAM OFDM), adjust LED drive currents, or switch beam steering angles.

### Feature Selection Matrix:

| Role | Variable | Engineering Notes | Physical Relevance in Car-to-Car VLC |
| :--- | :--- | :--- | :--- |
| **Input Feature 1** | `distancia` | Impute 0 values with median/interpolation | Governs fundamental geometric decay: $H_{\text{LOS}} \propto \frac{1}{d^2}$ |
| **Input Feature 2** | `ambientLight` | Log-transform or standardize | Determines optical shot noise and receiver SNR degradation |
| **Input Feature 3** | `angRadiacion` | Encode or scale | Models transmitter LED beam spread ($\phi$) |
| **Input Feature 4** | `angIncidencia` | Encode or scale | Models receiver photodetector alignment & effective area ($\psi$) |
| **Input Feature 5** | `presenciaPersonas`| Keep binary ($0/1$) | Models pedestrian shadowing / physical beam blockage |
| **Input Feature 6** | `entorno` | One-Hot Encode / Binary ($0/1$) | Captures multipath reflections from surroundings (lab vs glass) |
| **Target Variable** | `pathloss` | Replace `-inf` with channel cutoff (-65 dB) | Continuous regression target for channel sensing |

---

## 8. Verification Script

The complete Python script to verify all findings directly against `dataset.mat` is located at:
`scripts/analyze_dataset.py`

To execute:
```bash
python scripts/analyze_dataset.py
```
*(or run via `.venv/Scripts/python scripts/analyze_dataset.py`)*
