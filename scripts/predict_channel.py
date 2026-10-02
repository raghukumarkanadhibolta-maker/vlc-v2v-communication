#!/usr/bin/env python3
"""
scripts/predict_channel.py

Academic Project:
AI-BASED ADAPTIVE CAR-TO-CAR VISIBLE LIGHT COMMUNICATION WITH ML-BASED CHANNEL SENSING

Standalone Inference & Dynamic Link Adaptation Module
Loads the trained champion model (models/channel_quality_model.pkl) and evaluates
VLC channel conditions using purely physical pre-reception input features:
  - distance (meters)
  - ambientLight (lux)
  - angRadiacion (degrees)
  - angIncidencia (degrees)
  - presenciaPersonas (0 = clear, 1 = obstructed)
  - entorno (0/'laboratorio' or 1/'pasilloCristal')

Outputs exact required strings:
  Predicted Channel: GOOD or POOR
  Recommended Modulation: OOK or BSK + FEC
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import joblib
import pandas as pd
import numpy as np


def parse_entorno(val):
    if isinstance(val, (int, float, np.integer)):
        return int(val)
    val_str = str(val).strip().lower()
    if val_str in ['0', 'lab', 'laboratorio', 'laboratory']:
        return 0
    elif val_str in ['1', 'pasillo', 'pasillocristal', 'glass_corridor', 'corridor']:
        return 1
    else:
        raise ValueError(f"Unrecognized environment '{val}'. Use 'laboratorio' (0) or 'pasilloCristal' (1).")


def predict_channel(distance, ambient_light, ang_radiacion, ang_incidencia, presencia_personas, entorno, model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "channel_quality_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")

    # Load serialized pipeline
    model = joblib.load(model_path)

    # Standardize inputs
    dist_m = float(distance)
    # If distance was entered in cm (> 10), convert to meters
    if dist_m > 10.0:
        dist_m = dist_m / 100.0

    amb = float(ambient_light)
    rad_ang = float(ang_radiacion)
    inc_ang = float(ang_incidencia)
    personas = 1 if int(presencia_personas) > 0 else 0
    ent_code = parse_entorno(entorno)

    input_df = pd.DataFrame([{
        'distancia_m': dist_m,
        'ambientLight': amb,
        'angRadiacion': rad_ang,
        'angIncidencia': inc_ang,
        'presenciaPersonas': personas,
        'entorno_encoded': ent_code
    }])

    # Predict
    pred_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
    prob_poor = probabilities[0]
    prob_good = probabilities[1]

    # Map to communication action via 3-Mode Adaptive Transmission Logic
    if pred_class == 1:
        channel_str = "GOOD"
        modulation_str = "OOK"
        rationale = "High optical channel gain (PL >= -35 dB). Fast On-Off Keying selected for maximum throughput (up to 10-20 Mbps) with low processing overhead."
        ook_metrics = None
        bpsk_metrics = None
    else:
        channel_str = "POOR"
        from scripts.adaptive_vlc_demo import evaluate_adaptive_vlc
        eval_res = evaluate_adaptive_vlc(
            distance=dist_m,
            ambient_light=amb,
            ang_radiacion=rad_ang,
            ang_incidencia=inc_ang,
            presencia_personas=personas,
            entorno=ent_code,
            model_path=model_path,
            n_trials=30
        )
        modulation_str = eval_res['selected_modulation']
        rationale = eval_res['reason']
        ook_metrics = eval_res['ook_metrics']
        bpsk_metrics = eval_res['bpsk_metrics']

    return {
        'channel_quality': channel_str,
        'recommended_modulation': modulation_str,
        'class_code': pred_class,
        'prob_poor': prob_poor,
        'prob_good': prob_good,
        'rationale': rationale,
        'ook_metrics': ook_metrics,
        'bpsk_metrics': bpsk_metrics,
        'inputs': {
            'distance_m': dist_m,
            'ambientLight': amb,
            'angRadiacion': rad_ang,
            'angIncidencia': inc_ang,
            'presenciaPersonas': personas,
            'entorno': 'laboratorio' if ent_code == 0 else 'pasilloCristal'
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="VLC Channel Quality Prediction and Dynamic Modulation Recommendation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--distance", type=float, help="Propagation distance in meters (e.g., 0.8)")
    parser.add_argument("--ambientLight", type=float, help="Ambient optical illuminance in lux (e.g., 25.0)")
    parser.add_argument("--angRadiacion", type=float, help="Transmitter radiation angle in degrees (e.g., 5.0)")
    parser.add_argument("--angIncidencia", type=float, help="Receiver incidence angle in degrees (e.g., 10.0)")
    parser.add_argument("--presenciaPersonas", type=int, help="Pedestrian obstruction presence (0 = clear, 1 = blocked)")
    parser.add_argument("--entorno", type=str, help="Operating environment: 'laboratorio' (0) or 'pasilloCristal' (1)")
    parser.add_argument("--model", type=str, default=None, help="Path to channel_quality_model.pkl")

    # Positional fallback support
    parser.add_argument("pos_args", nargs="*", help="Optional positional inputs: distance ambientLight angRadiacion angIncidencia presenciaPersonas entorno")

    args = parser.parse_args()

    # Parse arguments
    dist = args.distance
    amb = args.ambientLight
    rad = args.angRadiacion
    inc = args.angIncidencia
    pers = args.presenciaPersonas
    ent = args.entorno

    if len(args.pos_args) == 6:
        dist, amb, rad, inc, pers, ent = (
            float(args.pos_args[0]),
            float(args.pos_args[1]),
            float(args.pos_args[2]),
            float(args.pos_args[3]),
            int(args.pos_args[4]),
            args.pos_args[5]
        )
    elif dist is None or amb is None or rad is None or inc is None or pers is None or ent is None:
        # If running interactively without all arguments, prompt or provide default
        if sys.stdin.isatty():
            print("\n--- Interactive VLC Channel Sensing Input ---")
            dist = float(input("Distance (m) [e.g. 0.85]: ") or 0.85)
            amb = float(input("Ambient Light (lux) [e.g. 30.0]: ") or 30.0)
            rad = float(input("Radiation Angle (deg) [e.g. 0.0]: ") or 0.0)
            inc = float(input("Incidence Angle (deg) [e.g. 5.0]: ") or 5.0)
            pers = int(input("Pedestrian Presence (0=No, 1=Yes) [e.g. 0]: ") or 0)
            ent = input("Environment (laboratorio / pasilloCristal) [e.g. laboratorio]: ") or "laboratorio"
        else:
            # Non-interactive default demonstration
            dist = 0.85
            amb = 30.0
            rad = 0.0
            inc = 5.0
            pers = 0
            ent = "laboratorio"
            print(f"[INFO] No CLI args provided; evaluating default scenario: distance={dist}m, ambientLight={amb}lx, angRadiacion={rad}°, angIncidencia={inc}°, presenciaPersonas={pers}, entorno='{ent}'")

    res = predict_channel(
        distance=dist,
        ambient_light=amb,
        ang_radiacion=rad,
        ang_incidencia=inc,
        presencia_personas=pers,
        entorno=ent,
        model_path=args.model
    )

    print("\n" + "=" * 65)
    print("VLC ADAPTIVE SENSING INFERENCE RESULTS")
    print("=" * 65)
    print(f"Inputs: Dist={res['inputs']['distance_m']:.2f}m | Amb={res['inputs']['ambientLight']:.1f}lx | RadAngle={res['inputs']['angRadiacion']:.1f} deg | IncAngle={res['inputs']['angIncidencia']:.1f} deg | Blocked={res['inputs']['presenciaPersonas']} | Env={res['inputs']['entorno']}")
    print(f"Confidence: P(Good) = {res['prob_good']*100:.1f}%, P(Poor) = {res['prob_poor']*100:.1f}%")
    print("-" * 65)
    print(f"Predicted Channel: {res['channel_quality']}")
    if res['channel_quality'] == 'GOOD':
        print(f"Selected Modulation: {res['recommended_modulation']}")
        print("-" * 65)
        print(f"Physical Rationale: {res['rationale']}")
    else:
        print("-" * 65)
        print("POOR CHANNEL MODULATION COMPARISON (Rate 1/2 Convolutional FEC):")
        ook = res['ook_metrics']
        bpsk = res['bpsk_metrics']
        print(f"  [OOK + FEC]  BER (Pre/Post): {ook['pre_ber']:.4f} / {ook['post_ber']:.6f} | Packet Success: {ook['fec_success_rate']*100:5.1f}% | Throughput: {ook['throughput_kbps']:5.1f} kbps | Latency: {ook['latency_ms']:.2f} ms")
        print(f"  [BPSK + FEC] BER (Pre/Post): {bpsk['pre_ber']:.4f} / {bpsk['post_ber']:.6f} | Packet Success: {bpsk['fec_success_rate']*100:5.1f}% | Throughput: {bpsk['throughput_kbps']:5.1f} kbps | Latency: {bpsk['latency_ms']:.2f} ms")
        print("-" * 65)
        print(f"Selected Modulation: {res['recommended_modulation']}")
        print(f"Reason for selection: {res['rationale']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
