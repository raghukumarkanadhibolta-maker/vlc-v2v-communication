# Final VLC Channel-Quality Target Validation Report
## Academic Project: AI-Based Adaptive Car-to-Car Visible Light Communication with ML-Based Channel Sensing

---

### Executive Summary

This report delivers the final pre-training scientific validation of the binary target variable within the verified ML-ready dataset [`dataset/ml_ready/vlc_channel_ml_ready.csv`](file:///c:/Users/raghu/OneDrive/Desktop/DVLC_VLC_ML_Animation/dataset/ml_ready/vlc_channel_ml_ready.csv).

#### Strict Research Integrity & Pre-Training Constraints:
1. **Dataset Invariance**: Neither the original raw dataset (`dataset.mat`) nor the ML-ready dataset (`dataset/ml_ready/vlc_channel_ml_ready.csv`) has been altered or modified in any way.
2. **Zero Model Training**: No machine learning model has been trained.
3. **Zero Target Leakage**: Confirmed 100% absence of received-signal-derived variables from the feature matrix $\mathbf{X}$.

---

## 1. Inspection of TARGET_VALIDATION_REPORT.md

The comprehensive findings in [`reports/TARGET_VALIDATION_REPORT.md`](file:///c:/Users/raghu/OneDrive/Desktop/DVLC_VLC_ML_Animation/reports/TARGET_VALIDATION_REPORT.md) have been thoroughly inspected and corroborated:
* The target variable is derived strictly from calibrated optical channel attenuation (path loss) rather than categorical environment labels.
* The assumption that `laboratorio = Good` and `pasilloCristal = Poor` was formally disproved: inside the laboratory alone, the target exhibits an exact $50.04\% : 49.96\%$ split (3,212 Poor vs 3,207 Good).
* Every physical sensor variable demonstrates statistically significant ($p < 10^{-20}$), physically consistent differences aligned with free-space optical wireless communication principles.

---

## 2. Exact Binary Target Definition

The ground-truth binary target is formulated as:

$$\mathbf{target\_channel\_quality\_binary} = \begin{cases} 
\mathbf{1\ (Good\ Channel)}, & \text{pathloss\_clean\_db} \ge -35.0\text{ dB} \\
\mathbf{0\ (Poor\ Channel)}, & \text{pathloss\_clean\_db} < -35.0\text{ dB}
\end{cases}$$

where `pathloss_clean_db` is the calibrated optical attenuation measured across the Thorlabs PDA100A2 amplified photodiode reception window, bounded to a physical hardware floor of $-60.0\text{ dB}$ during total line-of-sight outages ($V_{pp} = 0.0\text{ V}$).

---

## 3. Sample Counts & Exact Path Loss Statistics

The statistical breakdown of the 6,924 samples across both target classes is detailed below:

| Metric / Parameter | Overall Dataset ($N = 6,924$) | Class 0: Poor Channel ($< -35.0\text{ dB}$) | Class 1: Good Channel ($\ge -35.0\text{ dB}$) | Physical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Sample Count ($N$)** | **6,924 samples** | **3,591 samples** | **3,333 samples** | Complete preservation of all telemetry |
| **Class Percentage** | $100.00\%$ | **`51.86%`** | **`48.14%`** | Near-perfect 50/50 balance |
| **Path Loss Mean** | $-35.79\text{ dB}$ | **$-38.95\text{ dB}$** | **$-32.39\text{ dB}$** | Good channel collects $+6.56\text{ dB}$ higher optical signal |
| **Path Loss Std Dev** | $4.64\text{ dB}$ | $4.12\text{ dB}$ | $2.04\text{ dB}$ | Tight clustering in Good channel |
| **Path Loss Minimum** | $-60.00\text{ dB}$ | $-60.00\text{ dB}$ | $-34.98\text{ dB}$ | Absolute separation at $-35.0\text{ dB}$ |
| **Path Loss 25% ($Q_1$)** | $-37.45\text{ dB}$ | $-40.69\text{ dB}$ | $-34.20\text{ dB}$ | Lower quartile attenuation boundary |
| **Path Loss Median (50%)**| $-35.11\text{ dB}$ | $-37.40\text{ dB}$ | $-32.64\text{ dB}$ | Median path loss difference $= 4.76\text{ dB}$ |
| **Path Loss 75% ($Q_3$)** | $-32.81\text{ dB}$ | $-36.20\text{ dB}$ | $-30.95\text{ dB}$ | Upper quartile attenuation boundary |
| **Path Loss Maximum** | $-27.84\text{ dB}$ | $-35.003\text{ dB}$ | $-27.84\text{ dB}$ | Highest received optical signal |
| **Electrical SNR Mean** | $50.80\text{ dB}$ | **$44.42\text{ dB}$** | **$57.68\text{ dB}$** | Class 1 provides $+13.26\text{ dB}$ higher electrical SNR |
| **Optical Power $P_{rx}$ Mean**| $16.31\ \mu\text{W}$ | **$6.85\ \mu\text{W}$** | **$26.51\ \mu\text{W}$** | Class 1 collects nearly $4\times$ optical flux |
| **Bit Error Rate (BER)** | $2.92\cdot 10^{-4}$ | **$5.62\cdot 10^{-4}$** | **$0.00\cdot 10^{-0}$** | Error-free ($< 10^{-12}$) across all Class 1 samples |

### 3.1 Class Balance Assessment
* **Class Ratio (Poor / Good)**: $\frac{3,591}{3,333} = \mathbf{1.0774 : 1}$.
* **Shannon Target Entropy**:
  $$H(Y) = - \sum_{i \in \{0, 1\}} p_i \log_2(p_i) = -0.5186 \log_2(0.5186) - 0.4814 \log_2(0.4814) = \mathbf{0.9994\text{ bits}} \approx 1.0\text{ bit}$$
* **Verdict**: **SUFFICIENTLY BALANCED**. The classes are balanced within a $1.08 : 1$ ratio, eliminating algorithmic bias and precluding the need for artificial resampling (SMOTE) or class weighting.

### 3.2 Exact Source of the $-35.0\text{ dB}$ Threshold
1. **Empirical Sample Median**: The median path loss across all 6,924 samples is **$-35.11\text{ dB}$**. Rounding to $-35.0\text{ dB}$ establishes an information-theoretically balanced median split.
2. **Operational Inflection Point**: At $-35.0\text{ dB}$, received optical power transitions across $12.9\ \mu\text{W}$ ($P_{rx} \le 12.87\ \mu\text{W}$ in Class 0 vs $P_{rx} \ge 12.95\ \mu\text{W}$ in Class 1), and average electrical SNR shifts from high-fidelity ($57.68\text{ dB}$) down to degraded conditions ($44.42\text{ dB}$).
3. **Shadowing Sensitivity**: Unobstructed free-space measurements have a mean path loss of $-35.31\text{ dB}$, while pedestrian-shadowed measurements have a mean path loss of $-36.37\text{ dB}$. The $-35.0\text{ dB}$ boundary cleanly captures the onset of shadowing-induced fading.

---

## 4. Target Leakage Confirmation

An exhaustive programmatic scan of the feature matrix $\mathbf{X}$ confirmed that **zero received-signal-derived variables** exist in the predictor inputs:

| Prohibited Received-Signal Variable | Present in $\mathbf{X}$? | Audit Status |
| :--- | :---: | :---: |
| `pathloss_clean_db` | **NO** | **PASSED** |
| `pathloss` | **NO** | **PASSED** |
| `vpp_volts` | **NO** | **PASSED** |
| `vpp_counts` | **NO** | **PASSED** |
| `p_rx_uW` | **NO** | **PASSED** |
| `p_rx_dBm` | **NO** | **PASSED** |
| `p_amb_uW` | **NO** | **PASSED** |
| `snr_db` / `snr_linear` | **NO** | **PASSED** |
| `theoretical_ber_ook` / `log10_ber` | **NO** | **PASSED** |
| `is_pathloss_outage` | **NO** | **PASSED** |
| `i_sig_A` / `i_bg_A` | **NO** | **PASSED** |

> [!IMPORTANT]
> **LEAKAGE AUDIT VERDICT**: **ZERO TARGET LEAKAGE (0.00%)**.

---

## 5. Permitted Input Features (Available Prior to Reception)

The feature matrix $\mathbf{X}$ is composed strictly of 6 physical variables measurable prior to, or independently of, optical signal demodulation:

$$\mathbf{X} = [\text{distancia\_m}, \text{ambientLight}, \text{angRadiacion}, \text{angIncidencia}, \text{presenciaPersonas}, \text{entorno\_encoded}]$$

1. **`distancia_m`** ($0.007\text{ m} - 2.366\text{ m}$): Line-of-sight distance measured by the onboard HC-SR04 ultrasonic sonar sensor. Governs the free-space geometric decay law $H_{\text{LOS}} \propto \frac{1}{d^2}$.
2. **`ambientLight`** ($1.919\text{ lx} - 1,984.082\text{ lx}$): Background optical illuminance measured by the independent BH1750 ambient light lux meter. Governs receiver background shot noise variance $\sigma_{\text{shot}}^2 = 2 q I_{\text{bg}} B$.
3. **`angRadiacion`** ($\{0^\circ, 30^\circ, 45^\circ\}$): Transmitter LED emission angle ($\phi$) preset by vehicle headlight alignment or steering angle. Dictates Lambertian radiant intensity $R_0(\phi) = \frac{m+1}{2\pi}\cos^m(\phi)$.
4. **`angIncidencia`** ($\{0^\circ, 30^\circ, 45^\circ\}$): Receiver photodiode tilt angle ($\psi$) determined by receiver gimbal or vehicle orientation. Controls effective optical collection area $A_{\text{eff}}(\psi) = A_{\text{det}}\cos(\psi)$.
5. **`presenciaPersonas`** ($\{0, 1\}$): Binary obstacle/pedestrian flag detected by vehicle forward-facing ultrasonic sensors, radar, or cameras. Captures abrupt line-of-sight blockage and deep optical fading.
6. **`entorno_encoded`** ($\{0, 1\}$): Binary environmental classification ($0$ = `laboratorio`, $1$ = `pasilloCristal`), available via vehicle digital map context, GPS, or optical daylight sensors. Captures the extreme solar illuminance dynamic range ($10\times - 50\times$ increase in outdoor glass corridors).

---

## 6. Suitability Confirmation for Adaptive V2V VLC Transmission

```
                                  +------------------------------------+
                                  |     Vehicle Sensory Inputs (X)     |
                                  | (Distance, Ambient Light, Angles,  |
                                  |   Blockage Flag, Environment)      |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |      ML Channel-Quality Model      |
                                  |     (Predicts Channel State)       |
                                  +-----------------+------------------+
                                                    |
                               +--------------------+--------------------+
                               |                                         |
                               v                                         v
                    [P(Good) >= 0.5]                          [P(Good) < 0.5]
               +-------------------------+               +----------------------------+
               |  CLASS 1: GOOD CHANNEL  |               |   CLASS 0: POOR CHANNEL    |
               |  - Path Loss >= -35 dB  |               |   - Path Loss < -35 dB     |
               |  - SNR Mean = 57.7 dB   |               |   - SNR Mean = 44.4 dB     |
               |  - Optical Power > 13 uW|               |   - Optical Power < 13 uW  |
               +------------+------------+               +-------------+--------------+
                            |                                          |
                            v                                          v
               +-------------------------+               +----------------------------+
               |   COMMUNICATION ACTION  |               |    COMMUNICATION ACTION    |
               |       SELECT: OOK       |               |    SELECT: BSK + FEC       |
               | - Maximum Throughput    |               | - Subcarrier Shift Keying  |
               | - Minimum Latency       |               | - Forward Error Correction |
               | - Energy Efficient      |               | - 3-6 dB Coding Gain       |
               +-------------------------+               +----------------------------+
```

### Physical & Communication-Theoretic Alignment:
1. **Good Channel Regime $\implies$ Action: OOK (On-Off Keying)**:
   * **Channel State**: Mean $\text{Path Loss} = -32.39\text{ dB}$, Mean $\text{SNR} = 57.68\text{ dB}$, Mean $P_{rx} = 26.51\ \mu\text{W}$, $\text{BER} < 10^{-12}$.
   * **Communication Rationale**: On-Off Keying (Intensity Modulation / Direct Detection) provides maximum spectral throughput and zero decoding overhead. When SNR exceeds $50\text{ dB}$, un-coded OOK operates completely error-free. Applying FEC under these conditions would introduce unnecessary coding latency and waste transmission bandwidth.
2. **Poor Channel Regime $\implies$ Action: BSK + Forward Error Correction (FEC)**:
   * **Channel State**: Mean $\text{Path Loss} = -38.95\text{ dB}$, Mean $\text{SNR} = 44.42\text{ dB}$, Mean $P_{rx} = 6.85\ \mu\text{W}$, elevated shot noise and blockage fading.
   * **Communication Rationale**: When path loss drops below $-35.0\text{ dB}$, optical signal amplitude shrinks below $0.7\text{ V}$, making baseband OOK vulnerable to threshold drift, ambient solar shot noise, and pedestrian shadow fading.
   * **Why BSK + FEC**:
     * **BSK (Binary Subcarrier Shift Keying)**: Modulates the bitstream onto a high-frequency subcarrier, shifting the signal away from the low-frequency DC band where ambient solar glare and artificial lighting noise concentrate.
     * **FEC (Forward Error Correction)**: Provides an additional $3\text{ dB}$ to $6\text{ dB}$ of coding gain, enabling error recovery and packet integrity even during deep shadow fades.
3. **Conclusion**: The $-35.0\text{ dB}$ threshold provides an operationally valid switching boundary between un-coded/light OOK and robust BSK + FEC.

---

## 7. Pre-Training Final Verification Checklist

- [x] Full inspection of `reports/TARGET_VALIDATION_REPORT.md` completed.
- [x] Exact binary target definition verified ($\text{Good} \ge -35.0\text{ dB}$, $\text{Poor} < -35.0\text{ dB}$).
- [x] Sample counts and percentages reported ($N = 6,924$; Good = $3,333$ [$48.14\%$], Poor = $3,591$ [$51.86\%$]).
- [x] Complete descriptive statistics for path loss reported for both classes.
- [x] Class balance confirmed ($1.077 : 1$ ratio, entropy $H = 0.9994\text{ bits}$).
- [x] Exact source of $-35.0\text{ dB}$ threshold documented (empirical median split, operational power boundary, shadowing transition).
- [x] Complete absence of received-signal variables from feature matrix $\mathbf{X}$ confirmed (zero target leakage).
- [x] Permitted input features confirmed to be pre-reception vehicle sensory variables only.
- [x] Target confirmed to be operationally aligned with the intended adaptive transmission policy (Good $\to$ OOK, Poor $\to$ BSK + FEC).
- [x] Neither original dataset nor ML-ready dataset was modified.
- [x] Zero machine learning models were trained.

---

### Final Formal Declaration

TARGET_VALIDATION: APPROVED
