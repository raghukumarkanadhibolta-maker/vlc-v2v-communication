# Final Project Audit and Verification

**Project Title:** AI-Based Adaptive Car-to-Car Visible Light Communication with Machine Learning Channel Sensing  
**Audit Date:** September 28, 2026  
**Audit Result:** **ALL 11 COMPONENTS VERIFIED & FULLY FUNCTIONAL (100%)**  

---

## 1. Project Status
* **Overall Status:** **COMPLETED & FULLY VERIFIED**
* **Project Type:** Software Simulation, Physical-Layer Emulation, and Data-Driven Adaptive Decision Pipeline.

---

## 2. Component Verification Matrix

| Component | File Path | Status | Verification Summary |
|---|---|:---:|---|
| **1. Original Dataset** | `dataset/VLC-measurement-platform-main/.../dataset.mat` | **VERIFIED** | Original MATLAB experimental measurement data preserved untouched. |
| **2. ML-Ready Dataset** | `dataset/processed/vlc_channel_dataset_processed.csv` | **VERIFIED** | 6,926 cleaned, normalized rows without data leakage. |
| **3. ML Model** | `models/channel_quality_model.pkl` | **VERIFIED** | Champion Gradient Boosting Pipeline (96.1% test acc, 0.992 ROC-AUC). |
| **4. Inference Script** | `scripts/predict_channel.py` | **VERIFIED** | Standalone inference from 6 pre-reception physical inputs ($12.0\text{ }\mu\text{s}$). |
| **5. Adaptive 3-Mode Logic** | `scripts/adaptive_vlc_demo.py` | **VERIFIED** | GOOD $\rightarrow$ OOK; POOR $\rightarrow$ benchmark OOK+FEC vs BPSK+FEC. |
| **6. OOK + FEC Transceiver** | `scripts/vlc_transceiver_sim.py` | **VERIFIED** | Unipolar OOK + Rate 1/2 Convolutional Code + Soft Viterbi Decoding. |
| **7. BPSK + FEC Transceiver** | `scripts/vlc_transceiver_sim.py` | **VERIFIED** | SIM-BPSK Subcarrier + Rate 1/2 Convolutional Code + Soft Viterbi. |
| **8. Comparison Results** | `results/poor_channel_modulation_comparison.csv` | **VERIFIED** | 17-point SNR sweep ($0.0\text{ dB} - 16.0\text{ dB}$, 897,600 bits/modulation). |
| **9. Comparison Figures** | `reports/*.png` (6 figures) | **VERIFIED** | Publication-quality BER, PER, Tput, Latency, Success, Dashboard plots. |
| **10. Dual-Car Demo UI** | `scripts/adaptive_vlc_demo.py` & `reports/car_to_car_vlc_dashboard.html` | **VERIFIED** | CAR A (Tx) $\rightarrow$ Optical Link $\rightarrow$ CAR B (Rx Dashboard with Buzzer). |
| **11. Academic Reports** | `reports/FINAL_ACADEMIC_PROJECT_REPORT.md` | **VERIFIED** | Complete 21-section academic report with grounded numerical data. |

---

## 3. Verified Comparison Figures

All required figures are present in the `reports/` directory and referenced:
1. **Bit Error Rate (BER) vs SNR:** `reports/ber_ook_fec_vs_bpsk_fec.png`
2. **Packet Error Rate (PER) vs SNR:** `reports/poor_channel_per_comparison.png`
3. **Throughput vs SNR:** `reports/throughput_ook_fec_vs_bpsk_fec.png`
4. **Latency vs SNR:** `reports/latency_ook_fec_vs_bpsk_fec.png`
5. **FEC Success Rate vs SNR:** `reports/fec_success_ook_fec_vs_bpsk_fec.png`
6. **Overall 4-Panel Performance Dashboard:** `reports/poor_channel_overall_comparison.png`

---

## 4. Adaptive Decision Flow Verification

```
[ Pre-Reception Physical Parameters (d, Amb, θ, ψ, Blocked, Env) ]
                               │
                               ▼
            [ ML Channel Classifier: Gradient Boosting ]
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼ (GOOD Channel)                              ▼ (POOR Channel)
  [ Uncoded OOK ]                         [ Comparative Poor Benchmark ]
  - Throughput: 1000.0 kbps               - Simulate OOK+FEC vs BPSK+FEC
  - Latency: 0.53 ms                      - Target: Packet Success >= 95%
  - Zero FEC Overhead                     - SNR >= 5 dB: Select OOK+FEC (+15% Tput)
                                          - SNR < 5 dB: Select BPSK+FEC (Outage)
        │                                             │
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
     [ VLC Optical Link Simulation (Path Loss, Solar Shot Noise, AWGN) ]
                               │
                               ▼
        [ CAR B Receiver Dashboard (Demodulation, Viterbi, CRC) ]
                               │
                               ▼
               [ 🚨 Emergency Buzzer Alert Triggered ]
```

---

## 5. Commands to Launch the Final Demonstration

### Option 1: Terminal Interactive Demonstration (Recommended)
```bash
.\.venv\Scripts\python.exe scripts\adaptive_vlc_demo.py --interactive
```

### Option 2: Automated Batch Test of All 3 Scenarios
```bash
.\.venv\Scripts\python.exe scripts\adaptive_vlc_demo.py --run-all-scenarios
```

### Option 3: Individual Preset Scenarios
```bash
# Scenario 1 (GOOD channel -> OOK):
.\.venv\Scripts\python.exe scripts\adaptive_vlc_demo.py --scenario 1

# Scenario 2 (Moderate POOR channel -> OOK+FEC):
.\.venv\Scripts\python.exe scripts\adaptive_vlc_demo.py --scenario 2

# Scenario 3 (Severe POOR channel -> BPSK+FEC):
.\.venv\Scripts\python.exe scripts\adaptive_vlc_demo.py --scenario 3
```

### Option 4: Graphical Web Dashboard
Open `reports/car_to_car_vlc_dashboard.html` in any web browser to view the interactive two-vehicle visual interface, live sliders, animated light beams, and real-time Car B telemetry.

---

## 6. Software Simulation Scope & Hardware Distinctions
* **Nature of Project:** This project is a physics-based, data-driven software simulation and emulation pipeline.
* **Distinction from Real Hardware:**
  * No physical automotive LED headlights, photodiode hardware, or microcontrollers (STM32/ESP32) are claimed to be physically deployed.
  * Real-world automotive deployment will require analog optical front-end filters to suppress DC ambient sunlight and hardware AC coupling.
  * Optical transmission and error recovery are fully validated using mathematical channel physics, real experimental propagation measurements, and DSP algorithms.
