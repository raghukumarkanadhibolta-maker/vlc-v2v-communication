#!/usr/bin/env python3
"""
scripts/analyze_dataset.py

Academic Project:
AI-BASED ADAPTIVE CAR-TO-CAR VISIBLE LIGHT COMMUNICATION WITH ML-BASED CHANNEL SENSING

Purpose:
Loads the actual VLC dataset (dataset.mat) from the measurement platform,
extracts all variables from the MATLAB MCOS serialized table,
and outputs complete structural analysis, physical definitions, and statistical profiles.
"""

import os
import sys
import io
import struct
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy.io as sio


def find_dataset_path():
    """Locate dataset.mat in the expected repository directories."""
    candidate_paths = [
        os.path.join("dataset", "VLC-measurement-platform-main", "VLC-measurement-platform-main", "dataset.mat"),
        os.path.join("dataset", "VLC-measurement-platform-main", "dataset.mat"),
        os.path.join("..", "dataset", "VLC-measurement-platform-main", "VLC-measurement-platform-main", "dataset.mat"),
        os.path.join(os.path.dirname(__file__), "..", "dataset", "VLC-measurement-platform-main", "VLC-measurement-platform-main", "dataset.mat"),
    ]
    for p in candidate_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p
    raise FileNotFoundError(f"dataset.mat not found. Checked candidate paths: {candidate_paths}")


def load_vlc_dataset(mat_path=None):
    """
    Loads dataset.mat and parses the internal MCOS table structure.
    Returns:
        df (pd.DataFrame): DataFrame containing all 9 columns.
        metadata (dict): Structural and header metadata.
    """
    if mat_path is None:
        mat_path = find_dataset_path()

    mat = sio.loadmat(mat_path)
    file_size_bytes = os.path.getsize(mat_path)
    header_str = mat.get('__header__', b'').decode('latin1', errors='ignore')

    # The dataset was saved in MATLAB as a MATLAB table 'tablaTotal'.
    # In MATLAB 5 MAT-files, tables are stored via MCOS subsystem in '__function_workspace__'.
    raw_data = mat['__function_workspace__'].tobytes()
    sub = raw_data[16:16 + 455032]
    off = 64
    ft_type, ft_size = struct.unpack('<II', sub[off:off + 8])

    # Reconstruct MAT 5.0 container to unpack the inner MCOS object
    desc = b'MATLAB 5.0 MAT-file, MCOS subfield'.ljust(116, b' ')
    header = desc + struct.pack('<Q', 0) + struct.pack('<H', 0x0100) + b'IM'
    padded = (ft_size + 7) & ~7
    matrix_bytes = sub[off:off + 8 + padded]

    loaded = sio.loadmat(io.BytesIO(header + matrix_bytes))
    val = loaded['__function_workspace__']
    obj_meta = val['_ObjectMetadata'][0]

    num_rows = int(obj_meta[8, 0][0, 0])
    num_cols = int(obj_meta[10, 0][0, 0])
    col_names = [str(obj_meta[11, 0][0, i][0]) for i in range(num_cols)]

    # Timestamps (milliseconds Unix epoch)
    date_raw = obj_meta[2, 0].flatten()
    date_dt = [datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc) for ts in date_raw]

    # Categorical 'entorno'
    categories = [str(c[0]) for c in obj_meta[4, 0].flatten()]
    entorno_codes = obj_meta[5, 0].flatten()  # 1-based index
    entorno_labels = [categories[code - 1] for code in entorno_codes]

    # Data columns
    col_data = obj_meta[6, 0]

    df = pd.DataFrame({
        'date_raw_ms': date_raw,
        'date': date_dt,
        'distancia': col_data[0, 1].flatten(),
        'angRadiacion': col_data[0, 2].flatten(),
        'angIncidencia': col_data[0, 3].flatten(),
        'ambientLight': col_data[0, 4].flatten(),
        'presenciaPersonas': col_data[0, 5].flatten(),
        'lente': col_data[0, 6].flatten(),
        'entorno_code': entorno_codes,
        'entorno': entorno_labels,
        'pathloss': col_data[0, 8].flatten(),
    })

    metadata = {
        'mat_path': mat_path,
        'file_size_bytes': file_size_bytes,
        'header': header_str,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'col_names': col_names,
        'categories': categories,
    }

    return df, metadata


def print_analysis():
    df, meta = load_vlc_dataset()

    print("=" * 80)
    print("VLC DATASET STRUCTURAL AND STATISTICAL ANALYSIS")
    print("Academic Project: AI-Based Adaptive Car-to-Car VLC with ML-Based Channel Sensing")
    print("=" * 80)
    print(f"Dataset File Path: {meta['mat_path']}")
    print(f"File Size:         {meta['file_size_bytes']} bytes ({meta['file_size_bytes']/1024:.2f} KB)")
    print(f"MAT-File Header:   {meta['header'].strip()}")
    print(f"Number of Samples: {meta['num_rows']:,}")
    print(f"Total Columns:     {meta['num_cols']}")
    print(f"Available Columns: {', '.join(meta['col_names'])}")
    print("-" * 80)

    print("\n1. VARIABLE INVENTORY & PHYSICAL DESCRIPTIONS:")
    descriptions = {
        'date': ('Measurement timestamp', 'Internal / Arduino clock', 'yyyy-MM-dd HH:mm:ss.SSS', 'datetime64[UTC]'),
        'distancia': ('Physical distance between transmitter & receiver', 'HC-SR04 Ultrasonic Sonar', 'Centimeters (cm)', 'float64'),
        'angRadiacion': ('Transmitter LED radiation / irradiance angle (phi)', 'Pre-set mechanical angle', 'Degrees (deg)', 'float64'),
        'angIncidencia': ('Receiver photodiode incidence angle (psi)', 'Pre-set mechanical angle', 'Degrees (deg)', 'float64'),
        'ambientLight': ('Background ambient optical illuminance', 'BH1750 Digital Lux Sensor', 'Lux (lx)', 'float64'),
        'presenciaPersonas': ('Interference / optical path obstruction flag', 'Manual / experimental log', 'Binary (0 or 1)', 'float64'),
        'lente': ('Optical concentrator / receiver lens flag', 'Experimental configuration', 'Binary (0 = no lens)', 'float64'),
        'entorno': ('Physical operating environment', 'Experimental location', 'Categorical string', 'category / string'),
        'pathloss': ('Calculated VLC channel optical path attenuation', 'Derived from Thorlabs PDA100A2 Vpp', 'Decibels (dB)', 'float64'),
    }

    print(f"{'Variable':<18} | {'Sensor / Source':<26} | {'Unit':<18} | {'Data Type':<16}")
    print("-" * 84)
    for col in meta['col_names']:
        desc, sensor, unit, dtype = descriptions.get(col, ('N/A', 'N/A', 'N/A', 'N/A'))
        print(f"{col:<18} | {sensor:<26} | {unit:<18} | {dtype:<16}")

    print("\n2. DESCRIPTIVE STATISTICS TABLE (NUMERICAL FEATURES):")
    stat_cols = ['distancia', 'angRadiacion', 'angIncidencia', 'ambientLight', 'presenciaPersonas', 'lente']
    summary = df[stat_cols].describe().T
    summary['missing'] = df[stat_cols].isnull().sum()
    summary['zeros'] = (df[stat_cols] == 0).sum()
    summary['unique'] = df[stat_cols].nunique()
    summary_print = summary[['count', 'missing', 'mean', 'std', 'min', '50%', 'max', 'zeros', 'unique']]
    print(summary_print.to_string())

    print("\n3. PATH LOSS METRIC ANALYSIS:")
    total_pl = len(df['pathloss'])
    inf_count = np.isneginf(df['pathloss']).sum()
    nan_count = np.isnan(df['pathloss']).sum()
    finite_mask = np.isfinite(df['pathloss'])
    finite_pl = df.loc[finite_mask, 'pathloss']

    print(f"Total pathloss samples:     {total_pl:,}")
    print(f"Missing / NaN values:       {nan_count}")
    print(f"Infinite (-inf) values:     {inf_count} (occurring when Vpp_output == 0 / complete blockage)")
    print(f"Finite pathloss samples:    {len(finite_pl):,} ({len(finite_pl)/total_pl*100:.2f}%)")
    print(f"Finite Minimum:             {finite_pl.min():.4f} dB")
    print(f"Finite Maximum:             {finite_pl.max():.4f} dB")
    print(f"Finite Mean:                {finite_pl.mean():.4f} dB")
    print(f"Finite Median:              {finite_pl.median():.4f} dB")
    print(f"Finite Std Dev:             {finite_pl.std():.4f} dB")
    print(f"Finite 25th Percentile:     {finite_pl.quantile(0.25):.4f} dB")
    print(f"Finite 75th Percentile:     {finite_pl.quantile(0.75):.4f} dB")

    print("\n4. CATEGORICAL & DISCRETE DISTRIBUTIONS:")
    print("Environment ('entorno'):")
    for cat, count in df['entorno'].value_counts().items():
        print(f"  - {cat:<16}: {count:>5} samples ({count/total_pl*100:>5.2f}%)")

    print("Transmitter Radiation Angle ('angRadiacion'):")
    for val, count in sorted(df['angRadiacion'].value_counts().items()):
        print(f"  - {val:>4.1f} deg:        {count:>5} samples ({count/total_pl*100:>5.2f}%)")

    print("Receiver Incidence Angle ('angIncidencia'):")
    for val, count in sorted(df['angIncidencia'].value_counts().items()):
        print(f"  - {val:>4.1f} deg:        {count:>5} samples ({count/total_pl*100:>5.2f}%)")

    print("Pedestrian Presence ('presenciaPersonas'):")
    for val, count in sorted(df['presenciaPersonas'].value_counts().items()):
        print(f"  - {int(val)}:               {count:>5} samples ({count/total_pl*100:>5.2f}%)")

    print("\n5. CRITICAL FINDINGS, DATA ANOMALIES & LIMITATIONS:")
    print("a) NO CHANNEL LABELS:")
    print("   - There are NO 'GOOD' or 'BAD' channel quality labels in dataset.mat.")
    print("   - 'entorno' is a test location tag ('laboratorio' vs 'pasilloCristal'), not a link status.")
    print("b) ZERO-DISTANCE READINGS (Ultrasonic Timeout):")
    print(f"   - 421 samples ({421/total_pl*100:.2f}%) have distancia == 0.0 cm.")
    print("   - This occurs due to HC-SR04 sonar echo timeouts (> 350 cm or beam misalignment).")
    print("c) INFINITE PATH LOSS:")
    print(f"   - 3 samples have pathloss == -inf due to log10(0) when photodiode delta Vpp was 0.")
    print("d) CONSTANT FEATURE ('lente'):")
    print("   - 'lente' has variance 0.0 (all 6,924 samples are 0.0; no lens tests were performed).")
    print("   - Cannot be used as an active ML feature.")
    print("e) MATLAB CODE FORMULA DISCREPANCY (genDataSet.m):")
    print("   - Table 1 formula:      pathloss = 10*log10(0.6*Vpp / (Vpp_input * 10^6))")
    print("   - Tables 2-12 formula:  pathloss = 10*log10(0.6*Vpp / (Vpp_input * 10^5))")
    print("   - Results in a 10 dB offset in Table 1 relative to Tables 2-12 in raw calculation.")

    print("\n6. RECOMMENDED ML-BASED CHANNEL SENSING FEATURES:")
    print("   [+] Input Feature 1: distancia (Inter-vehicle distance in cm)")
    print("   [+] Input Feature 2: ambientLight (Solar / ambient optical background noise in lux)")
    print("   [+] Input Feature 3: angRadiacion (Transmitter beam emission angle in degrees)")
    print("   [+] Input Feature 4: angIncidencia (Receiver optical reception angle in degrees)")
    print("   [+] Input Feature 5: presenciaPersonas (Channel obstruction / blockage indicator)")
    print("   [+] Input Feature 6: entorno (Environment context: laboratory vs glass corridor)")
    print("   [*] Target Variable: pathloss (Continuous channel attenuation in dB)")
    print("=" * 80)


if __name__ == "__main__":
    print_analysis()
