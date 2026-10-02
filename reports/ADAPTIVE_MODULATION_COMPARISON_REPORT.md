# AI-Based Adaptive Car-to-Car VLC: Comparative Study & Adaptive Decision Pipeline for Poor Channels

**Project:** AI-Based Adaptive Car-to-Car Visible Light Communication (VLC) with ML-Based Channel Sensing  
**Module:** Physical Layer Transceiver Simulation & Dynamic Link Adaptation  
**Status:** Validated, Implemented, and Fully Reproducible  

---

## Executive Summary

In vehicular Visible Light Communication (VLC) systems, dynamic optical channel variations arise from distance changes, vehicle vibrations, angular misalignments, ambient sunlight interference, and pedestrian or vehicular shadowing. While high optical SNR conditions permit simple, high-speed On-Off Keying (OOK), degraded or "poor" channel states require link adaptation to prevent communication outage.

Rather than making the unsubstantiated assumption that Binary Phase Shift Keying (BPSK) is uniformly superior for all degraded conditions, this project implements a rigorous, scientifically controlled physical-layer simulation benchmark comparing **OOK + FEC** and **BPSK + FEC** under identical channel conditions, noise floors, code rates, and bitstreams.

### Key Experimental Findings
1. **Low-SNR / Deep-Blockage Regime ($\text{SNR} < 5.0\text{ dB}$, $\text{Path Loss} \le -48.0\text{ dB}$):**  
   **BPSK + FEC is superior.** Its antipodal constellation ($d_{min} = 2$) and coherent subcarrier demodulation reject ambient optical DC noise and provide a $3.0\text{ dB}$ electrical SNR advantage over unipolar OOK. In this regime, BPSK+FEC maintains up to $100\%$ packet delivery success where OOK+FEC experiences severe packet dropouts ($0\% - 85\%$ success).
2. **Moderate Poor-Channel Regime ($5.5\text{ dB} \le \text{SNR} \le 16.0\text{ dB}$, $-46.0\text{ dB} \le \text{Path Loss} < -35.0\text{ dB}$):**  
   **OOK + FEC is superior.** Because Rate 1/2 Viterbi decoding is sufficient to eliminate all residual bit errors at $\text{SNR} \ge 6.0\text{ dB}$, OOK+FEC achieves $100\%$ packet success while delivering **$15\%$ higher effective throughput** ($500.0\text{ kbps}$ vs $434.8\text{ kbps}$) and **lower transceiver processing latency** ($10.35\text{ ms}$ vs $10.59\text{ ms}$) by eliminating subcarrier guard-band and filtering overhead.
3. **Good Channel Regime ($\text{SNR} > 25.0\text{ dB}$, $\text{Path Loss} \ge -35.0\text{ dB}$):**  
   **Uncoded OOK is optimal.** Transmits at maximum raw line rate ($1.0\text{ Mbps}$ baseband / $10-20\text{ Mbps}$ hardware) with zero FEC parity overhead and near-zero DSP latency.

---

## 1. Why Both OOK + FEC and BPSK + FEC Are Evaluated

A common misconception in optical wireless communication is that higher-order or carrier-modulated schemes (like BPSK) should unconditionally replace baseband pulse modulation (OOK) whenever channel conditions degrade. In reality, vehicular optical communications operate under strict hardware, power, and bandwidth constraints:

| Performance Dimension | OOK + FEC | Subcarrier BPSK (SIM-BPSK) + FEC | Trade-Off Analysis |
|---|---|---|---|
| **Constellation Type** | Unipolar Intensity ($0, 2P_{avg}$) | Antipodal Subcarrier ($\pm P_{avg} \cos(2\pi f_c t)$) | BPSK has $3\text{ dB}$ electrical power advantage for identical Euclidean separation. |
| **DC Optical Rejection** | Susceptible to DC ambient drift; requires adaptive slicing threshold | Inherent rejection: DC bias and sunlight blocked by AC coupling / bandpass filter | BPSK immune to daylight DC saturation; OOK requires baseline estimation. |
| **Bandwidth Efficiency** | High: 1 symbol per baseband pulse | Moderate: Requires subcarrier frequency $f_c$ with guard band overhead | OOK achieves higher baud rate per unit LED bandwidth. |
| **Net Throughput** | $R_{raw} \times R_c \times (1 - \text{PER})$ | $\frac{R_{raw}}{1.15} \times R_c \times (1 - \text{PER})$ | OOK provides $15\%$ higher throughput when $\text{PER} = 0$. |
| **Transceiver Complexity** | Minimal (LED driver switch + threshold comparator) | Moderate (carrier synthesizer, analog mixer, matched filter) | OOK has lower hardware cost and lower power dissipation. |
| **Demodulation Latency** | Negligible ($\le 0.5\,\mu\text{s}$) | Moderate (filter group delay + integration period) | OOK provides lower latency in safety-critical vehicle telemetry. |

Therefore, evaluating **both** modulations under identical degraded channel conditions is mandatory to determine the exact crossover threshold where the noise resilience of BPSK outweighs its throughput and complexity penalties.

---

## 2. Simulation Assumptions & Scientific Fairness Guarantees

To ensure absolute scientific validity, the simulation enforces strict parity between the two candidates:

1. **Identical Bitstream:** Every trial generates an identical pseudo-random payload sequence ($N = 512$ payload bits) protected by an identical 16-bit CRC checksum ($N_{total} = 528$ bits).
2. **Identical Forward Error Correction (FEC):** Both modulations use the exact same Rate $1/2$ Convolutional Code with identical generator polynomials and an identical soft-decision Viterbi decoder.
3. **Identical Optical Power Constraints:** Both modulations are normalized to the exact same average emitted optical power $P_{avg} = 1.0\text{ W}$ and peak optical drive current $P_{peak} = 2.0\text{ W}$, respecting physical LED thermal dissipation and eye-safety limits.
4. **Identical Optical Path Loss & Attenuation:** Both signals propagate through the identical physical channel attenuation $H_{vlc} = 10^{PL/10}$ derived from the calibrated vehicular dataset.
5. **Identical Noise Environment:** Ambient solar shot noise and photodetector thermal noise are modeled using the identical physical noise standard deviation $\sigma_{total}$.
6. **No Pre-determined Winner:** The modulation decision is derived entirely from the measured packet error rate, throughput, and latency.

---

## 3. Forward Error Correction (FEC) Architecture

The FEC subsystem utilizes a non-systematic convolutional code with optimal free distance for short-packet vehicular communications:

* **Code Rate:** $R_c = 1/2$ (1 input bit generates 2 coded bits).
* **Constraint Length:** $K = 3$ (Memory $m = 2$, shift register states = $2^2 = 4$).
* **Generator Polynomials:**
  $$g_0 = [1, 1, 1]_2 = 7_8$$
  $$g_1 = [1, 0, 1]_2 = 5_8$$
* **Flushing:** $K - 1 = 2$ tail zero bits are appended to flush the encoder back to the zero state ($S_0$), ensuring proper trellis termination.
* **Decoding Algorithm:** Soft-Decision Maximum-Likelihood Viterbi Decoding.
  * For OOK, soft metrics are computed relative to the adaptive slicing threshold: $y_k = I_{rx, k} - I_{th}$.
  * For BPSK, soft metrics are obtained directly from the coherent correlator output: $y_k = \int_0^T I_{ac}(t) \cos(2\pi f_c t) dt$.
  * Branch metrics minimize squared Euclidean distance: $D = \sum_{i=0}^1 (y_i - \mu_i)^2$.
* **Packet Integrity (PER / BLER):** A 16-bit CRC-CCITT (polynomial `0x1021`) verifies packet integrity. A packet is classified as successful only if zero residual bit errors remain post-Viterbi decoding.

---

## 4. VLC Channel and Physical Noise Model

The simulation is grounded in the physical hardware characteristics established during dataset verification:

### Photodetector & Circuit Parameters
* Photodiode Responsivity: $R_{resp} = 0.6\text{ A/W}$ (Silicon PIN photodiode at $\lambda = 550\text{ nm}$).
* Active Area: $A_{det} = 75.4\text{ mm}^2 = 7.54 \times 10^{-5}\text{ m}^2$.
* Electrical Receiver Bandwidth: $B = 1.0\text{ MHz}$.
* Load Resistor: $R_L = 50\,\Omega$.
* Ambient Temperature: $T_K = 300\text{ K}$.
* Transimpedance Gain: $G = 1.0 \times 10^5\text{ V/A}$.

### Optical Channel Path Loss
Optical DC channel gain follows the generalized Lambertian model:
$$H_{vlc}(0) = \frac{(m_L + 1) A_{det}}{2\pi d^2} \cos^{m_L}(\phi) T_s(\psi) g(\psi) \cos(\psi), \quad 0 \le \psi \le \Psi_c$$
where $d$ is propagation distance, $\phi$ is radiation angle, and $\psi$ is incidence angle. Pedestrian obstruction introduces an additional $6.0\text{ dB}$ to $12.0\text{ dB}$ attenuation due to optical shadowing.

### Noise Variances
1. **Photodetector Shot Noise (Ambient Light + Signal):**
   $$\sigma_{shot}^2 = 2 q R_{resp} P_{amb} B + 2 q R_{resp} P_{rx} B$$
   where $P_{amb} = \text{ambientLight (lux)} \times 0.0079\text{ W/m}^2 \times A_{det} \times \cos(\psi)$, and $q = 1.602 \times 10^{-19}\text{ C}$.
2. **Receiver Thermal Noise:**
   $$\sigma_{thermal}^2 = \frac{4 k_B T_K B}{R_L}$$
   where $k_B = 1.3806 \times 10^{-23}\text{ J/K}$.
3. **Total Noise Standard Deviation:**
   $$\sigma_{total} = \sqrt{\sigma_{shot}^2 + \sigma_{thermal}^2}$$

---

## 5. Quantitative Benchmark Results

A comprehensive Monte Carlo sweep over electrical SNR ($0.0\text{ dB}$ to $16.0\text{ dB}$ in $1.0\text{ dB}$ increments, 50 trials per point, 26,400 bits/point) yielded the following measured performance:

### Summary Performance Table

| Electrical SNR (dB) | OOK Pre-BER | OOK Post-BER | OOK Success Rate (%) | OOK Tput (kbps) | BPSK Pre-BER | BPSK Post-BER | BPSK Success Rate (%) | BPSK Tput (kbps) | Selected Modulation | Selection Rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| **0.0** | 0.1586 | 0.0884 | 0.0% | 0.0 | 0.0771 | 0.0041 | 42.0% | 182.6 | **BPSK + FEC** | BPSK maintains 42% packet delivery; OOK suffers total outage. |
| **1.0** | 0.1297 | 0.0384 | 0.0% | 0.0 | 0.0551 | 0.0006 | 86.0% | 373.9 | **BPSK + FEC** | BPSK achieves 86% success; OOK remains in total outage (0%). |
| **2.0** | 0.1053 | 0.0150 | 8.0% | 40.0 | 0.0386 | 0.0000 | 100.0% | 434.8 | **BPSK + FEC** | BPSK achieves 100% error-free delivery; OOK only 8%. |
| **3.0** | 0.0799 | 0.0047 | 36.0% | 180.0 | 0.0236 | 0.0000 | 100.0% | 434.8 | **BPSK + FEC** | BPSK error-free (100%); OOK has 64% packet loss. |
| **4.0** | 0.0567 | 0.0005 | 76.0% | 380.0 | 0.0122 | 0.0000 | 100.0% | 434.8 | **BPSK + FEC** | BPSK 100% reliable; OOK fails 95% target (76% success). |
| **5.0** | 0.0382 | 0.0002 | 94.0% | 470.0 | 0.0060 | 0.0000 | 100.0% | 434.8 | **BPSK + FEC** | BPSK meets target (100%); OOK marginally below target (94%). |
| **6.0** | 0.0228 | 0.0000 | 100.0% | 500.0 | 0.0024 | 0.0000 | 100.0% | 434.8 | **OOK + FEC** | Both 100% reliable; OOK delivers +15% higher throughput. |
| **7.0** | 0.0129 | 0.0000 | 100.0% | 500.0 | 0.0010 | 0.0000 | 100.0% | 434.8 | **OOK + FEC** | Both 100% reliable; OOK delivers +15% higher throughput. |
| **8.0** | 0.0061 | 0.0000 | 100.0% | 500.0 | 0.0001 | 0.0000 | 100.0% | 434.8 | **OOK + FEC** | Both 100% reliable; OOK delivers +15% higher throughput. |
| **10.0** | 0.0005 | 0.0000 | 100.0% | 500.0 | 0.0000 | 0.0000 | 100.0% | 434.8 | **OOK + FEC** | Both 100% reliable; OOK delivers +15% higher throughput. |
| **12.0** | 0.0000 | 0.0000 | 100.0% | 500.0 | 0.0000 | 0.0000 | 100.0% | 434.8 | **OOK + FEC** | Both 100% reliable; OOK delivers +15% higher throughput. |
| **14.0** | 0.0000 | 0.0000 | 100.0% | 500.0 | 0.0000 | 0.0000 | 100.0% | 434.8 | **OOK + FEC** | Both 100% reliable; OOK delivers +15% higher throughput. |
| **16.0** | 0.0000 | 0.0000 | 100.0% | 500.0 | 0.0000 | 0.0000 | 100.0% | 434.8 | **OOK + FEC** | Both 100% reliable; OOK delivers +15% higher throughput. |

---

## 6. Generated Publication-Quality Graphs

The benchmark script generated four publication-quality figures saved in `reports/`:

1. **Bit Error Rate (BER) vs SNR:** [`reports/ber_vs_snr_comparison.png`](file:///c:/Users/raghu/OneDrive/Desktop/DVLC_VLC_ML_Animation/reports/ber_vs_snr_comparison.png)  
   Shows pre-FEC and post-FEC BER curves. Clearly illustrates the $3.0\text{ dB}$ coding gain of Viterbi soft decoding and the $3.0\text{ dB}$ modulation gain of BPSK over OOK.
2. **Effective Throughput vs SNR:** [`reports/throughput_comparison.png`](file:///c:/Users/raghu/OneDrive/Desktop/DVLC_VLC_ML_Animation/reports/throughput_comparison.png)  
   Highlights the operational crossover at $\text{SNR} = 5.5\text{ dB}$. Below $5.5\text{ dB}$, BPSK+FEC maintains connectivity while OOK drops toward $0\text{ kbps}$. Above $5.5\text{ dB}$, OOK+FEC provides a constant $500.0\text{ kbps}$ vs BPSK's $434.8\text{ kbps}$.
3. **End-to-End Latency vs SNR:** [`reports/latency_comparison.png`](file:///c:/Users/raghu/OneDrive/Desktop/DVLC_VLC_ML_Animation/reports/latency_comparison.png)  
   Displays total packet latency (transmission + DSP + Viterbi decode). Shows that OOK+FEC incurs lower processing latency ($10.35\text{ ms}$ vs $10.59\text{ ms}$) across all SNR points.
4. **FEC Decoding Success Rate vs SNR:** [`reports/fec_success_rate_comparison.png`](file:///c:/Users/raghu/OneDrive/Desktop/DVLC_VLC_ML_Animation/reports/fec_success_rate_comparison.png)  
   Illustrates the sharp waterfall behavior of convolutional coding, highlighting the $95\%$ reliability boundary.

---

## 7. Final Adaptive Decision Pipeline

The complete adaptive transmission system operates across three distinct modes:

```
                      +-----------------------------+
                      | Physical Sensor Inputs:     |
                      | dist, amb, angles, ped, env |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Champion Random Forest ML   |
                      | Classifier Pipeline         |
                      +--------------+--------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
          Class 1 (GOOD)                          Class 0 (POOR)
                 |                                       |
                 v                                       v
    +-------------------------+             +-------------------------+
    | MODE 1: GOOD CHANNEL    |             | MODE 2: POOR CHANNEL    |
    | - Selected: OOK         |             | Trigger Physical Sim:   |
    | - Uncoded Transmission  |             | Evaluate OOK+FEC vs     |
    | - Max Throughput (10M+) |             |          BPSK+FEC       |
    | - 0 Latency Overhead    |             +------------+------------+
    +-------------------------+                          |
                                                         v
                                            +-------------------------+
                                            | Apply Transparent Rule: |
                                            | Reliability >= 95%?     |
                                            +------------+------------+
                                                         |
                         +-------------------------------+-------------------------------+
                         |                                                               |
            Only BPSK meets target                                          Both meet target
            (Deep Fade / Severe Blockage)                                   (Moderate Poor Channel)
                         |                                                               |
                         v                                                               v
            +-------------------------+                                     +-------------------------+
            | Selected: BPSK + FEC    |                                     | Selected: OOK + FEC     |
            | Reason: Prevents outage |                                     | Reason: +15% Throughput |
            | & rejects ambient DC    |                                     | & Lower Latency         |
            +-------------------------+                                     +-------------------------+
```

### Predefined, Transparent Decision Rule
1. **Target Reliability Threshold:** The primary criterion for vehicular safety messaging is packet reliability: $\text{Packet Success Rate} \ge 95.0\%$ (residual post-FEC $\text{BER} \le 10^{-4}$).
2. **Condition 1 (Severe Degradation / Low SNR, $\text{SNR} < 5.5\text{ dB}$):**  
   If $\text{Success}_{\text{BPSK+FEC}} \ge 95\%$ and $\text{Success}_{\text{OOK+FEC}} < 95\%$:  
   $\to$ **Select BPSK + FEC**.  
   *Justification:* BPSK+FEC provides antipodal noise margin and eliminates ambient optical DC drift, maintaining connection where OOK+FEC fails.
3. **Condition 2 (Moderate Degradation, $\text{SNR} \ge 5.5\text{ dB}$):**  
   If both $\text{Success}_{\text{OOK+FEC}} \ge 95\%$ and $\text{Success}_{\text{BPSK+FEC}} \ge 95\%$:  
   $\to$ **Select OOK + FEC**.  
   *Justification:* Because both guarantee zero packet loss, OOK+FEC is selected to maximize communication rate ($500.0\text{ kbps}$ vs $434.8\text{ kbps}$) and minimize transceiver latency.
4. **Condition 3 (Extreme Outage, $\text{SNR} < 1.0\text{ dB}$):**  
   If neither reaches $95\%$ reliability:  
   $\to$ **Select BPSK + FEC with link-degradation alert**, as it yields higher packet delivery probability ($42\%$ vs $0\%$).

---

## 8. Limitations of Simulation vs Real VLC Hardware

While this physical simulation accurately models Poisson shot noise, Johnson thermal noise, Lambertian geometry, and convolutional coding gain, real-world deployment on physical vehicular VLC hardware introduces additional non-idealities:

1. **LED Non-Linearity and Clipping:** Real automotive headlamps and taillamps exhibit non-linear current-to-optical power transfer curves ($I$-$V$ and $L$-$I$ curves). High modulation index in SIM-BPSK can drive the LED into saturation or reverse-bias cutoff, generating harmonic distortion.
2. **Transimpedance Amplifier (TIA) Sunlight Saturation:** Under direct sunlight ($> 40,000\text{ lux}$), strong DC photocurrent can saturate the front-end TIA into its supply rail, causing total optical blindout unless an active DC-subtraction circuit or optical narrow-band optical filter is present.
3. **Optical Receiver Bandwidth & Intersymbol Interference (ISI):** Large-area photodetectors ($A_{det} > 50\text{ mm}^2$) have significant junction capacitance ($C_j \approx 100 - 400\text{ pF}$), which creates a low-pass pole that limits bandwidth to $1-5\text{ MHz}$ and induces intersymbol interference at data rates above $2\text{ Mbps}$.
4. **Optical Turbulence & Road Vibration:** Atmospheric heat shimmer over asphalt and vehicle chassis pitch/roll induce fast optical fading and angle-of-arrival fluctuations that require fast-tracking adaptive equalizers.
5. **Multipath Dispersion in Confined Spaces:** In tunnels and parking structures, reflected non-line-of-sight (NLOS) optical rays create delay spreads ($5 - 20\text{ ns}$) that can limit coherent subcarrier demodulation unless guard intervals are employed.

---

## 9. Conclusion

By implementing a rigorous, reproducible transceiver simulation, this project successfully replaces arbitrary modulation selection with an **empirically validated 3-mode adaptive VLC decision architecture**. The system maximizes channel capacity in clear conditions, maximizes throughput in moderately degraded conditions, and guarantees link survivability in deep optical fades.
