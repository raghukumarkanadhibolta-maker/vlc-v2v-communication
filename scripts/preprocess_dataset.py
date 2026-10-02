#!/usr/bin/env python3
"""
scripts/preprocess_dataset.py

Academic Project:
AI-BASED ADAPTIVE CAR-TO-CAR VISIBLE LIGHT COMMUNICATION WITH ML-BASED CHANNEL SENSING

Step 2: Dataset Preprocessing & VLC Physical Feature Engineering
- Loads original dataset.mat (READ-ONLY, preserves original file).
- Cleans invalid values:
  * Imputes ultrasonic sonar dropouts (distancia == 0 cm) via temporal session-based interpolation.
  * Handles -inf path loss caused by complete LOS blockage (clipped to -60.0 dB hardware floor).
  * Removes zero-variance 'lente' feature (constant 0 across all 6,924 samples).
- Engineers physical VLC channel features:
  * Reconstructed photodiode peak-to-peak voltage (Vpp in counts and Volts).
  * Received optical power (P_rx in uW and dBm).
  * Ambient background optical noise power and photodetector shot noise.
  * Electrical Signal-to-Noise Ratio (SNR in dB).
  * Theoretical Bit Error Rate (BER for OOK modulation).
  * Lambertian geometric channel factor (cos(phi) * cos(psi) / d^2).
- Defines Ground-Truth Channel Quality Targets:
  * Binary Target: Good Channel (1) vs Poor Channel (0) at -35.0 dB cutoff.
  * Multi-Class Target: Good (2) vs Moderate (1) vs Poor (0).
  * Continuous Target: Calibrated Path Loss (dB).
- Saves processed dataset separately in CSV and Parquet formats under dataset/processed/.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.special import erfc

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.analyze_dataset import load_vlc_dataset


def preprocess_vlc_dataset():
    print("=" * 80)
    print("STEP 2: PREPARING VLC DATASET FOR MACHINE LEARNING")
    print("=" * 80)

    # 1. Load original dataset without modification
    print("\n[Stage 1] Loading original dataset.mat (Read-Only)...")
    df_raw, meta = load_vlc_dataset()
    n_samples = len(df_raw)
    print(f"Loaded {n_samples:,} samples across {meta['num_cols']} raw features.")

    # 2. Detect Sessions via Inter-measurement Time Gaps
    print("\n[Stage 2] Detecting measurement sessions & time series structure...")
    df = df_raw.copy()
    df['dt'] = pd.to_datetime(df['date_raw_ms'], unit='ms')
    time_diff = df['dt'].diff().dt.total_seconds()
    # A gap > 10 seconds indicates a transition between experimental tables (sessions)
    df['session_id'] = (time_diff > 10).cumsum() + 1
    num_sessions = df['session_id'].nunique()
    print(f"Identified {num_sessions} discrete experimental sessions.")

    # 3. Handle Invalid Values & Outliers
    print("\n[Stage 3] Cleaning invalid values and handling dropouts...")

    # Diagnostic flags for transparency
    df['is_distance_dropout'] = (df['distancia'] == 0.0).astype(int)
    df['is_pathloss_outage'] = np.isneginf(df['pathloss']).astype(int)

    num_dist_dropouts = df['is_distance_dropout'].sum()
    num_pl_outages = df['is_pathloss_outage'].sum()
    print(f" - Ultrasonic distance dropouts (distancia == 0.0 cm): {num_dist_dropouts} ({num_dist_dropouts/n_samples*100:.2f}%)")
    print(f" - Path loss -inf singularities (Vpp == 0 / complete blockage): {num_pl_outages} ({num_pl_outages/n_samples*100:.2f}%)")

    # Impute distancia == 0 within each session using temporal linear interpolation + edge bfill/ffill
    df['distancia_imputed_cm'] = df.groupby('session_id')['distancia'].transform(
        lambda s: s.replace(0, np.nan).interpolate(method='linear').bfill().ffill()
    )
    df['distancia_m'] = df['distancia_imputed_cm'] / 100.0

    # Handle -inf in pathloss: replace with -60.0 dB (physical 1-bit quantization floor)
    df['pathloss_clean_db'] = df['pathloss'].replace(-np.inf, -60.0)

    # Remove zero-variance 'lente' feature
    print(" - Dropping zero-variance feature: 'lente' (all 6,924 samples are 0.0)")

    # 4. Feature Engineering: Physics-Based VLC Channel Metrics
    print("\n[Stage 4] Synthesizing physical VLC channel metrics...")

    # 4.1 Reconstruct Peak-to-Peak Signal Voltage (Vpp)
    # In genDataSet.m, divisor was 3e6 for Session 1 and 3e5 for Sessions 2-12
    divisors = np.where(df['session_id'] == 1, 3.0 * 1e6, 3.0 * 1e5)
    # Vpp (ADC counts, 0 - 1023)
    df['vpp_counts'] = np.where(
        df['is_pathloss_outage'] == 1,
        0.0,
        (divisors / 0.6) * (10.0 ** (df['pathloss'] / 10.0))
    )
    # Convert ADC counts to Volts (Arduino 10-bit ADC reference: 5.0V / 1023)
    df['vpp_volts'] = df['vpp_counts'] * (5.0 / 1023.0)

    # 4.2 Optical Received Power (P_rx)
    # Thorlabs PDA100A2 transimpedance gain (G): 1e6 V/A (Session 1), 1e5 V/A (Sessions 2-12)
    G_trans = np.where(df['session_id'] == 1, 1.0 * 1e6, 1.0 * 1e5)
    R_resp = 0.6  # Standard Si photodiode responsivity at visible spectrum (A/W)
    df['i_sig_A'] = df['vpp_volts'] / G_trans
    df['p_rx_W'] = df['i_sig_A'] / R_resp
    df['p_rx_uW'] = df['p_rx_W'] * 1e6
    df['p_rx_dBm'] = 10.0 * np.log10(np.maximum(df['p_rx_W'], 1e-12) / 1e-3)

    # 4.3 Ambient Optical Background Noise & Photodetector Noise
    # Photodiode active area = 75.4 mm^2 = 7.54e-5 m^2
    A_det = 7.54e-5
    # Daylight/white LED conversion factor: 1 lux approx 0.0079 W/m^2
    psi_rad = np.radians(df['angIncidencia'])
    phi_rad = np.radians(df['angRadiacion'])
    cos_psi = np.maximum(np.cos(psi_rad), 0.05)
    cos_phi = np.maximum(np.cos(phi_rad), 0.05)

    p_amb_W = df['ambientLight'] * 0.0079 * A_det * cos_psi
    df['p_amb_uW'] = p_amb_W * 1e6
    i_bg_A = R_resp * p_amb_W

    # Noise variance calculations
    q = 1.602176634e-19  # Electron charge (C)
    B = 1.0e6           # Receiver electrical bandwidth (1 MHz)
    k_B = 1.380649e-23   # Boltzmann constant (J/K)
    T_k = 300.0          # Absolute temperature (300 K)
    R_L = 50.0           # Load resistance (50 Ohms)

    sigma_shot2 = 2.0 * q * i_bg_A * B
    sigma_thermal2 = (4.0 * k_B * T_k * B) / R_L
    sigma_total2 = sigma_shot2 + sigma_thermal2

    # 4.4 Electrical Signal-to-Noise Ratio (SNR)
    df['snr_linear'] = (df['i_sig_A'] ** 2) / np.maximum(sigma_total2, 1e-25)
    df['snr_db'] = 10.0 * np.log10(np.maximum(df['snr_linear'], 1e-4))

    # 4.5 Theoretical Bit Error Rate (BER for OOK Modulation in AWGN)
    # BER = 0.5 * erfc(sqrt(SNR) / (2 * sqrt(2)))
    snr_eff = np.maximum(df['snr_linear'], 0.0)
    df['theoretical_ber_ook'] = 0.5 * erfc(np.sqrt(snr_eff) / (2.0 * np.sqrt(2.0)))
    df['log10_ber'] = np.log10(np.maximum(df['theoretical_ber_ook'], 1e-12))

    # 4.6 Geometric Optical Coupling & Line-of-Sight Alignment Factors
    df['cos_angRadiacion'] = cos_phi
    df['cos_angIncidencia'] = cos_psi
    df['angular_alignment_factor'] = cos_phi * cos_psi
    # Geometric path loss factor: H_LOS proportional to cos(phi)*cos(psi) / d^2
    df['geometric_loss_factor'] = df['angular_alignment_factor'] / (df['distancia_m'] ** 2)

    # 4.7 Log-Transformed Ambient Light (Addresses extreme skew from sunlight)
    df['log_ambientLight'] = np.log10(df['ambientLight'] + 1.0)

    # 4.8 Environment Encoding
    df['environment_id'] = (df['entorno'] == 'pasilloCristal').astype(int)

    # 5. Define Channel Quality Ground Truth Targets
    print("\n[Stage 5] Formulating Channel Quality Targets (Good vs Poor)...")
    # Binary Classification Target:
    # Good Channel (1): Path Loss >= -35.0 dB (High SNR > 50 dB, low attenuation, high throughput)
    # Poor Channel (0): Path Loss < -35.0 dB (High attenuation, low SNR, deep fade, or blockage)
    df['channel_quality_binary'] = (df['pathloss_clean_db'] >= -35.0).astype(int)

    # Multi-Class Classification Target:
    # 2 = Good (>= -34.0 dB)
    # 1 = Moderate (-38.0 dB <= PL < -34.0 dB)
    # 0 = Poor (< -38.0 dB)
    conditions = [
        df['pathloss_clean_db'] >= -34.0,
        (df['pathloss_clean_db'] < -34.0) & (df['pathloss_clean_db'] >= -38.0),
        df['pathloss_clean_db'] < -38.0,
    ]
    choices = [2, 1, 0]  # 2: Good, 1: Moderate, 0: Poor
    df['channel_quality_multi'] = np.select(conditions, choices, default=0)

    # Class distribution report
    b_counts = df['channel_quality_binary'].value_counts()
    print(" - Binary Channel Quality Target Distribution:")
    print(f"   * Poor Channel (0): {b_counts[0]:,} samples ({b_counts[0]/n_samples*100:.2f}%)")
    print(f"   * Good Channel (1): {b_counts[1]:,} samples ({b_counts[1]/n_samples*100:.2f}%)")

    m_counts = df['channel_quality_multi'].value_counts()
    print(" - Multi-Class Channel Quality Target Distribution:")
    print(f"   * Class 2 (Good):     {m_counts.get(2, 0):,} samples ({m_counts.get(2, 0)/n_samples*100:.2f}%)")
    print(f"   * Class 1 (Moderate): {m_counts.get(1, 0):,} samples ({m_counts.get(1, 0)/n_samples*100:.2f}%)")
    print(f"   * Class 0 (Poor):     {m_counts.get(0, 0):,} samples ({m_counts.get(0, 0)/n_samples*100:.2f}%)")

    # 6. Assemble Final Processed Feature Matrix
    print("\n[Stage 6] Assembling final processed DataFrame...")
    final_columns = [
        # Session and temporal metadata
        'session_id', 'date_raw_ms', 'dt',
        # Primary channel sensing inputs (Available to vehicle sensors)
        'distancia_imputed_cm', 'distancia_m',
        'ambientLight', 'log_ambientLight',
        'angRadiacion', 'cos_angRadiacion',
        'angIncidencia', 'cos_angIncidencia',
        'angular_alignment_factor', 'geometric_loss_factor',
        'presenciaPersonas',
        'entorno', 'environment_id',
        # Derived physical reception metrics
        'vpp_counts', 'vpp_volts',
        'p_rx_uW', 'p_rx_dBm',
        'p_amb_uW',
        'snr_db',
        'theoretical_ber_ook', 'log10_ber',
        # Quality Targets
        'pathloss_clean_db',
        'channel_quality_binary',
        'channel_quality_multi',
        # Diagnostic flags
        'is_distance_dropout', 'is_pathloss_outage',
    ]

    df_processed = df[final_columns].copy()

    # 7. Save Processed Dataset Separately
    output_dir = os.path.join("dataset", "processed")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "vlc_channel_dataset_processed.csv")
    parquet_path = os.path.join(output_dir, "vlc_channel_dataset_processed.parquet")

    print(f"\n[Stage 7] Saving processed dataset...")
    df_processed.to_csv(csv_path, index=False)
    print(f" [+] CSV saved to:     {os.path.abspath(csv_path)}")

    try:
        df_processed.to_parquet(parquet_path, index=False)
        print(f" [+] Parquet saved to: {os.path.abspath(parquet_path)}")
    except Exception as e:
        print(f" [!] Parquet export note: {e}")

    # Summary table of numeric features
    print("\n[Stage 8] Summary Statistics of Processed Dataset:")
    stat_summary = df_processed[[
        'distancia_imputed_cm', 'ambientLight', 'angRadiacion', 'angIncidencia',
        'presenciaPersonas', 'vpp_volts', 'p_rx_uW', 'snr_db', 'pathloss_clean_db',
        'channel_quality_binary'
    ]].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    print(stat_summary.to_string())

    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE: Processed dataset is ready for ML model training.")
    print("=" * 80)

    return df_processed


if __name__ == "__main__":
    preprocess_vlc_dataset()
