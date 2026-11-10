# Vehicle-to-Vehicle Li-Fi Communication System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/raghukumarkanadhibolta-maker/vlc-v2v-communication)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Baseline**: *"Vehicle to Vehicle Communication Using Li-Fi"* (2025 International Conference on Communication, Security and Privacy — ICCSP).  
> **Advanced Extension**: Production-grade physical-layer simulation, subcarrier intensity modulation (`SIM-QPSK + FEC`), 3-way modulation comparison (`OOK + FEC`, `BPSK + FEC`, `QPSK + FEC`), atmospheric weather modeling, vehicle vibration jitter, multi-hop relay forwarding, interactive highway animation, 13 dynamic research graphs, and live telemetry CSV export.

---

## 🌟 System Overview & Research Context

Visible Light Communication (VLC) / Li-Fi operates within the 400–700 nm optical band using automotive LED headlights and taillights as transmitters paired with silicon PIN photodiode receivers. Unlike RF systems (DSRC/C-V2X), Li-Fi is immune to electromagnetic interference, offers zero spectrum licensing fees, and delivers ultra-low physical transmission latency (< 15 ms).

### 1. Three Supported Modulation Schemes (100% Fair FEC Comparison)
All three schemes share the exact same **Rate 1/2 Convolutional Forward Error Correction (FEC)** code ($K=3$, generators $[7, 5]_8$, 4 trellis states) and **CRC-16 CCITT** frame validation:

| Modulation Scheme | Optical Implementation Architecture | Bits / Symbol | Theoretical Spectral Efficiency | Effective Throughput | Primary Advantage |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **OOK + FEC** | Unipolar On-Off Keying Intensity Modulation | 1 | 0.435 bps/Hz | 500.0 kbps | Lowest hardware complexity, direct LED switching |
| **BPSK + FEC** | Subcarrier Intensity Modulation (SIM-BPSK) with DC Bias | 1 | 0.435 bps/Hz | 434.8 kbps | Outstanding ambient DC-glare rejection, high noise margin (+3 dB) |
| **QPSK + FEC** | Subcarrier Intensity Modulation (SIM-QPSK) with DC Bias | 2 | 0.870 bps/Hz | 869.6 kbps | Double spectral efficiency & double net throughput |

### 2. Optical IM/DD SIM-QPSK Formulation
Because optical LEDs emit non-negative intensity ($P_{opt}(t) \ge 0$), bipolar electrical QPSK symbols cannot be directly transmitted. This project implements **Subcarrier Intensity Modulation (SIM-QPSK)** with DC bias:
$$P_{opt}(t) = P_{avg} \left[ 1 + m \left( I_k \cos(2\pi f_c t) - Q_k \sin(2\pi f_c t) \right) \right]$$
- Gray-coded symbol pairs $(I_k, Q_k) \in \{\pm \frac{1}{\sqrt{2}}\} \times \{\pm \frac{1}{\sqrt{2}}\}$ with average symbol energy $E_s = 1.0$.
- Modulation index $m \le 0.5$ guarantees $P_{opt}(t) \ge 0$ (no optical clipping).
- Receiver down-converts with coherent quadrature subcarrier recovery, rejecting ambient DC sunlight glare in the RF bandpass mixer.
- Soft Euclidean branch metrics are fed into the soft-decision Viterbi decoder for optimal maximum-likelihood sequence estimation.

---

## 🌦️ Real-World Environmental & Traffic Scenarios

1. **Clear Weather / Line of Sight**: 18.0 dB SNR at 1 m, 8.4 dB at 5 m, 4.2 dB at 10 m (calibrated to IEEE 2025 ICCSP reported measurements).
2. **Sunlight & Daytime**: Up to 85,000 lux background illuminance. Injects photodetector shot noise $\sigma_{shot} = \sqrt{2 q R_{pd} P_{amb} B_{rx}}$, reducing effective SNR by 3.5–5.0 dB.
3. **Rain Conditions**: Light ($1.2\text{ dB/m}$), Moderate ($2.8\text{ dB/m}$), and Heavy rain ($5.4\text{ dB/m}$) modeled using Carbonneau-Mie optical scattering.
4. **Fog Conditions**: Light ($2.5\text{ dB/m}$), Moderate ($6.0\text{ dB/m}$), and Heavy fog ($11.5\text{ dB/m}$) modeled via Kruse-Kim particle diffusion.
5. **Night-Time Transmission**: Low ambient illuminance (< 2.0 lux), headlight beam transmission, minimum photodiode noise floor.
6. **Vehicle Speed Scenarios (0 to 120 km/h)**: Models roadway vibration and chassis pitch jitter:
   $$\sigma_{\theta} = 0.05 \cdot \frac{v}{60}\text{ rad}$$
   Induces geometric alignment loss $L_{align} = \cos^m(\theta_{jitter})$.
7. **Road Hazards & Emergency Alerts**:
   - **Pothole Ahead**: Car A generates emergency message: `"POTHOLE AHEAD - SLOW DOWN"`.
   - **Animal Ahead**: Car A generates emergency message: `"ANIMAL AHEAD - REDUCE SPEED"`.
8. **Line-of-Sight Blockage & Multi-Hop Relay**:
   - Intermediate obstacle vehicle blocks direct optical LOS ($-25\text{ dB}$ optical attenuation).
   - Multi-hop relay mode dynamically activates Car B as an optical repeater forwarding to Car C (total latency = $2 \times t_{hop} + 1.2\text{ ms}$).

---

## 📊 Comprehensive 3-Way Comparative Evaluation

All parameters are computed by the live physical Monte Carlo simulation engine. When 0 bit errors are observed over $N$ bits, the **Poisson Rule of Three** statistical upper bound is reported ($BER < 3/N$ at 95% confidence):

| Performance Parameter | OOK + FEC | BPSK + FEC | QPSK + FEC | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| **BER before FEC** | $1.8 \times 10^{-3}$ | $8.2 \times 10^{-4}$ | $1.4 \times 10^{-3}$ | **BPSK + FEC** (lowest raw error) |
| **BER after FEC** | $< 1.15 \times 10^{-4}$ | $< 1.15 \times 10^{-4}$ | $< 1.15 \times 10^{-4}$ | **All 3 PASS** (Zero errors detected) |
| **Net Throughput** | 500.0 kbps | 434.8 kbps | **869.6 kbps** | **QPSK + FEC** ($2\times$ throughput) |
| **Transmission Latency** | **0.53 ms** | 10.65 ms | 5.45 ms | **OOK + FEC** (lowest latency) |
| **Electrical SNR** | 18.2 dB | 21.2 dB (+3 dB margin) | 18.2 dB | **BPSK + FEC** (+3 dB noise margin) |
| **Packet Delivery Ratio** | 100.0% | 100.0% | 100.0% | **All 3 Tied** (100.0% safety compliant) |
| **Spectral Efficiency** | 0.435 bps/Hz | 0.435 bps/Hz | **0.870 bps/Hz** | **QPSK + FEC** ($2\times$ spectral efficiency) |
| **Reliability Rating** | OPTIMAL | OPTIMAL | OPTIMAL | **BPSK (Adverse) / QPSK (Clear)** |

---

## 📈 13 Dynamic Research Analysis Charts

The web interface features an interactive Scientific Chart Studio supporting 13 parameter sweeps:
1. **BER vs. SNR**: Waterfall curves for OOK, BPSK, and QPSK before and after Rate 1/2 Viterbi decoding.
2. **BER Before FEC vs. BER After FEC**: Demonstrates +4.5 to +6.0 dB coding gain across all schemes.
3. **Throughput Comparison**: 3-bar comparative analysis across modulation types.
4. **Latency Comparison**: Total transmission time including serialization, propagation, and Viterbi traceback.
5. **SNR Comparison**: Electrical SNR under varying ambient noise levels.
6. **Throughput vs. Distance**: Net bit rate from 0.5 m to 10.0 m.
7. **BER vs. Distance**: Bit error escalation as inverse square path loss accumulates.
8. **BER vs. Vehicle Speed**: Geometric misalignment degradation from 0 to 120 km/h.
9. **Throughput vs. Vehicle Speed**: Speed-dependent packet loss impact on effective data rate.
10. **Performance under Different Weather Conditions**: Clear vs. Light/Mod/Heavy Rain vs. Light/Mod/Heavy Fog.
11. **Performance under Sunlight, Rain, Fog, and Night**: 4-quadrant environmental cross-comparison.
12. **Packet Delivery Ratio (PDR) Comparison**: Percentage of CRC-16 validated packets delivered.
13. **Spectral Efficiency Comparison**: Information transmission rate per unit optical bandwidth (bps/Hz).

---

## 💻 Interactive Highway Dashboard Features

* **3-Car Animated Highway Scene**: Visualizes Car A (Transmitter), Car B (Receiver / Relay), and Car C (Following Vehicle) with active LED headlights, optical cone beams, and road hazards.
* **Atmospheric Weather Overlay**: Real-time canvas particle simulation rendering rain streaks, fog particle diffusion, bright sunlight glare, and night darkness.
* **QPSK Constellation Diagram**: Live complex I/Q constellation display showing 4 symbol centroids with Gaussian noise cloud dispersion matching current channel SNR.
* **Car B Emergency Buzzer & Warning Strobe**: Integrated dual-tone acoustic alarm (960 Hz / 1280 Hz) using the Web Audio API with manual toggle and flashing roof strobe.
* **10-Step Physical Pipeline Tracker**: Live animation tracking data generation, FEC encoding, optical pulsing, channel propagation, photodiode reception, Viterbi decoding, CRC verification, and buzzer triggering.
* **CSV Telemetry Export**: One-click download of all simulation comparison metrics and historical runs via `POST /api/export-csv`.

---

## 🚀 Quickstart & Deployment

### Local Development
```bash
# Clone the repository
git clone https://github.com/raghukumarkanadhibolta-maker/vlc-v2v-communication.git
cd vlc-v2v-communication

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` in your web browser.

### Cloudflare Global Edge Tunnel (Public HTTPS Prototyping)
```powershell
# Run Cloudflare Tunnel pointing to local server
cloudflared tunnel --url http://127.0.0.1:8000
```

### Production Deployment (Render / Docker)
The repository includes a ready-to-use `render.yaml` blueprint:
1. Fork or push to your GitHub repository.
2. Link repository to [Render](https://render.com).
3. Automatic deployment will start with `pip install -r requirements.txt` and `uvicorn app:app --host 0.0.0.0 --port $PORT`.

---

## 📚 Academic References

1. **Base Paper**: *"Vehicle to Vehicle Communication Using Li-Fi"*, 2025 International Conference on Communication, Security and Privacy (ICCSP).
2. **IEEE 802.15.7 Standard**: *IEEE Standard for Local and Metropolitan Area Networks—Part 15.7: Short-Range Optical Wireless Communications*.
3. **Optical Channel Models**:
   - J. M. Kahn and J. R. Barry, *"Wireless Infrared Communications"*, Proceedings of the IEEE, 1997.
   - P. Carbonneau et al., *"Optical wireless communication channel modeling under atmospheric rain and fog"*, Applied Optics.
