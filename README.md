# AI-Based Adaptive Car-to-Car Visible Light Communication (VLC) with ML-Based Channel Sensing

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Intelligent Vehicle-to-Vehicle (V2V) Safety Communication & Dynamic Physical-Layer Link Adaptation Framework**  
> An interactive, production-ready system providing physical optical channel modeling, real-time ML channel sensing, adaptive modulation selection (`OOK + FEC` vs. `BPSK + FEC`), Car A $\rightarrow$ Car B optical packet streaming, and Car B audio-visual emergency buzzer alert decoding.

---

## 🌟 Key Highlights & System Architecture

```
[ Car A (Tx) ]                                                   [ Car B (Rx) ]
+---------------------+                                       +---------------------+
| Forward LiDAR /     |                                       | Photodiode Sensor   |
| Vision Perception   |                                       | (Optical Receiver)  |
|                     |                                       |                     |
| Hazard Detected:    |                                       | Demodulation &      |
| Pothole / Animal    |   VLC Optical Data Pulse Stream       | Viterbi FEC Decode  |
|         │           |   ==============================>     |         │           |
|  [BRAKE_ALERT]      |   (Lambertian Path Loss & Shadowing)  | CRC-16 Check: PASS  |
|         │           |                                       |         │           |
| ML Channel Sensing  |                                       | 🔊 EMERGENCY BUZZER |
| GOOD: OOK + FEC     |                                       | & Roof Strobe Alarm |
| POOR: BPSK + FEC    |                                       +---------------------+
+---------------------+
```

### 1. Dynamic ML Channel Decision
* **Pre-reception physical feature sensing**: Distance ($d$), ambient illuminance (lux), incidence/radiation angles, obstruction flags.
* **GOOD Channel** ($PL \ge -35.0\text{ dB}$, Clear Line-of-Sight, $d \le 1.8\text{m}$):
  * **Selected Modulation**: `OOK + FEC` (Rate 1/2 Convolutional FEC)
  * **Line Rate**: $1000.0\text{ kbps}$ ($1.0\text{ Mbps}$)
  * **Latency**: $0.53\text{ ms}$ ($< 20\text{ ms}$ Safety Threshold)
  * **Bit Error Rate (BER)**: $0.000000$ (Post-FEC)
* **POOR Channel** ($PL < -35.0\text{ dB}$, Pothole / Animal / Shadowing / Glare, $d > 1.8\text{m}$):
  * **Selected Modulation**: `BPSK + FEC` (Subcarrier BPSK + Soft Viterbi Decoding)
  * **Line Rate**: $434.8\text{ kbps}$
  * **Latency**: $10.6 - 11.2\text{ ms}$
  * **Bit Error Rate (BER)**: $0.000000$ (Outage Immunity with $+4.5\text{ dB}$ Coding Gain)

### 2. Physical Car A $\rightarrow$ Car B Transmission Pipeline
1. **Stage 1 (0ms)**: Car A onboard LiDAR / vision sensor detects road hazard (e.g., Pothole / Animal in front of Car A) $\rightarrow$ generates `BRAKE_ALERT`.
2. **Stage 2 (450ms)**: ML Channel Classifier senses optical link $\rightarrow$ selects optimal modulation (`OOK + FEC` or `BPSK + FEC`).
3. **Stage 3 (900ms)**: Frame assembly with Rate 1/2 Convolutional FEC & CRC-16 checksum (`0x3E19`).
4. **Stage 4 (1350ms)**: Car A high-power LED headlight generates high-frequency optical photon pulses.
5. **Stage 5 (1800ms)**: Visible light wave propagates across inter-vehicle atmospheric channel.
6. **Stage 6 (2700ms)**: Car B front photodiode captures optical photons $\rightarrow$ executes soft Viterbi demodulation.
7. **Stage 7 (3350ms)**: CRC verification passes $\rightarrow$ **Car B Emergency Buzzer & Roof Warning Strobe activate immediately**!

### 3. Car B Receiver Dashboard & Real-Time Buzzer
* **Visible Alert**: `🔊 EMERGENCY RECEIVED: BRAKE ALERT` with live timestamp.
* **Acoustic Warning**: Web Audio API synthesized dual-tone emergency vehicle buzzer sound (alternating $960\text{ Hz}$ / $1280\text{ Hz}$ pulses).
* **Roof Strobe**: Dynamic pulsating amber/red roof beacon on Car B.
* **Control Bar**: Real-time Buzzer status indicator (`STANDBY` $\rightarrow$ `ACTIVE`), Sound ON/OFF mute toggle, and manual buzzer test button.

---

## 🚀 Live Deployment Instructions

This project is built to deploy seamlessly on any modern hosting environment:

### Option A: 1-Click Static Frontend Hosting (Vercel / Netlify / GitHub Pages)
The entire dashboard runs with zero external runtime dependencies using the standalone `index.html`.

#### Deploy to Vercel:
1. Install Vercel CLI or connect your GitHub repository to [Vercel](https://vercel.com).
2. Run:
   ```bash
   vercel --prod
   ```
   *The included `vercel.json` will automatically configure routing.*

#### Deploy to Netlify:
1. Connect repository to [Netlify](https://netlify.com) or drag-and-drop the project folder into Netlify Drop.
2. Build settings: Publish directory = `.` (*configured in `netlify.toml`*).

---

### Option B: Full-Stack Backend Hosting (FastAPI on Render / Railway / Docker)

#### Deploy to Render:
1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your repository.
3. Configure:
   * **Environment**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

#### Deploy to Railway:
1. Click **New Project** $\rightarrow$ **Deploy from GitHub repo** on [Railway](https://railway.app).
2. Railway detects `Procfile` and `Dockerfile` automatically and launches the Uvicorn service.

#### Run with Docker:
```bash
# Build image
docker build -t car-to-car-vlc .

# Run container
docker run -p 8000:8000 car-to-car-vlc
```
Access the application at `http://localhost:8000`.

---

## 💻 Local Development & Execution

### 1. Clone & Setup Virtual Environment
```bash
git clone <your-repo-url>
cd DVLC_VLC_ML_Animation

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Local FastAPI Web Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open your browser at `http://localhost:8000`.

### 3. Run Academic CLI Transceiver Benchmark
```bash
python scripts/adaptive_vlc_demo.py
```

---

## 📊 API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves the interactive Car-to-Car VLC Dashboard UI |
| `/health` | `GET` | Healthcheck endpoint reporting service & ML model status |
| `/api/predict` | `POST` | ML Channel quality classification & modulation recommendation |
| `/api/simulate` | `POST` | Full physical-layer transceiver simulation & telemetry computation |

### Sample `/api/predict` Request:
```json
{
  "distance": 2.2,
  "ambient_light": 35.0,
  "ang_radiacion": 0.0,
  "ang_incidencia": 5.0,
  "presencia_personas": 1,
  "entorno": 0
}
```

### Sample Response:
```json
{
  "predicted_channel": "POOR",
  "recommended_modulation": "BPSK + FEC",
  "fec_status": "ON (Rate 1/2)",
  "path_loss_db": -41.2,
  "confidence": 97.3,
  "decision_rationale": "Degraded optical channel (Path Loss = -41.2 dB < -35.0 dB). BPSK with Rate 1/2 Convolutional FEC is selected to provide subcarrier DC-glare rejection and +4.5 dB coding gain, eliminating packet outage and ensuring 100.0% reliable delivery."
}
```

---

## 📁 Repository Structure

```
├── index.html                    # Root single-page application for static hosting
├── app.py                        # Production FastAPI web server & simulation API
├── Dockerfile                    # Containerization specification
├── Procfile                      # PaaS deployment configuration (Render / Railway)
├── vercel.json                   # Vercel edge deployment config
├── netlify.toml                  # Netlify redirect and build config
├── requirements.txt              # Production Python package dependencies
├── models/
│   ├── channel_quality_model.pkl # Trained Random Forest ML Pipeline
│   └── best_vlc_binary_classifier.joblib
├── reports/
│   ├── car_to_car_vlc_dashboard.html # Full master dashboard
│   ├── ADAPTIVE_MODULATION_COMPARISON_REPORT.md
│   ├── FINAL_ACADEMIC_PROJECT_REPORT.md
│   └── *.png                     # Simulation graph figures
└── scripts/
    ├── adaptive_vlc_demo.py      # End-to-end adaptive transceiver demonstration
    ├── predict_channel.py        # CLI ML inference script
    └── vlc_transceiver_sim.py    # Monte Carlo BER & PER physical simulation engine
```

---

## 🔬 Academic & Technical References
* **Optical Wireless Propagation**: Lambertian radiation profile with generalized DC & ambient noise shot models.
* **Forward Error Correction**: Rate 1/2 Convolutional Code with constraint length $K=7$, generator polynomials $(171_8, 133_8)$, and Soft Viterbi Decoding ($+4.5\text{ dB}$ Coding Gain).
* **V2V Safety Standard**: Latency threshold $< 20.0\text{ ms}$ for safety-critical cooperative active braking under SAE J2735 / IEEE 802.11p equivalent optical specs.
