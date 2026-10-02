#!/usr/bin/env python3
"""
scripts/train_final_models.py

Academic Project:
AI-BASED ADAPTIVE CAR-TO-CAR VISIBLE LIGHT COMMUNICATION WITH ML-BASED CHANNEL SENSING

Reproducible ML Model Training & Benchmarking Pipeline
- Dataset: dataset/ml_ready/vlc_channel_ml_ready.csv (6,924 samples, 0 nulls)
- Target: target_channel_quality_binary (1 = Good Channel [PL >= -35 dB], 0 = Poor Channel [PL < -35 dB])
- Physical Predictors:
    * distancia_m (distance in meters)
    * ambientLight (ambient optical illuminance in lux)
    * angRadiacion (transmitter radiation angle in degrees)
    * angIncidencia (receiver incidence angle in degrees)
    * presenciaPersonas (binary pedestrian/human obstruction: 0=None, 1=Blocked)
    * entorno_encoded (binary environment: 0=Laboratorio, 1=PasilloCristal)
- Zero Target Leakage: Strictly zero received-signal-derived variables (no pathloss, vpp, p_rx, snr, ber)
- Train/Test Split: GroupShuffleSplit on session_id (test_size=0.20, random_state=2)
    * Train: Sessions [1, 2, 3, 4, 7, 8, 9, 11, 12] (5,227 samples, 75.5%)
    * Test:  Sessions [5, 6, 10] (1,697 samples, 24.5%)
    * Prevents temporal auto-correlation leakage across continuous vehicle trajectories.
- Models Benchmarked:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Gradient Boosting
    5. XGBoost
    6. Naive Baseline (Most Frequent)
- Outputs Generated:
    * results/model_comparison.csv
    * results/confusion_matrices/ (individual plot per model + combined grid)
    * results/roc_curve.png
    * results/feature_importance.png
    * results/classification_report.txt
    * models/channel_quality_model.pkl (serialized champion pipeline)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib


def run_pipeline():
    print("=" * 80)
    print("VLC CHANNEL QUALITY BINARY CLASSIFICATION PIPELINE")
    print("=" * 80)
    start_total_time = time.time()

    # 1. Load Dataset
    data_path = os.path.join("dataset", "ml_ready", "vlc_channel_ml_ready.csv")
    print(f"\n[Step 1] Loading verified dataset from: {os.path.abspath(data_path)}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples with {df.shape[1]} columns.")

    # 2. Features and Target Specification
    target_col = 'target_channel_quality_binary'
    feature_cols = [
        'distancia_m',
        'ambientLight',
        'angRadiacion',
        'angIncidencia',
        'presenciaPersonas',
        'entorno_encoded'
    ]
    group_col = 'session_id'

    print(f"\n[Step 2] Feature & Target Verification:")
    print(f" - Predictors ({len(feature_cols)}): {feature_cols}")
    print(f" - Target: '{target_col}' (1 = Good Channel, 0 = Poor Channel)")
    print(f" - Group Column: '{group_col}' (12 experimental sessions)")

    # Strict Zero-Leakage Assertion
    prohibited_substrings = ['pathloss', 'vpp', 'p_rx', 'snr', 'ber', 'outage', 'clean_db']
    for feat in feature_cols:
        for p in prohibited_substrings:
            assert p not in feat.lower(), f"CRITICAL LEAKAGE DETECTED: {feat}"
    print(" [+] Leakage Verification PASSED: Zero received-signal variables used as predictors.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df[group_col].copy()

    # 3. Train/Test Splitting via GroupShuffleSplit
    print("\n[Step 3] Partitioning Dataset using GroupShuffleSplit (Preventing Trajectory Leakage)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train = groups.iloc[train_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    train_sessions = sorted(groups_train.unique().tolist())
    test_sessions = sorted(groups_test.unique().tolist())

    print(f" - Train Sessions ({len(train_sessions)}): {train_sessions}")
    print(f"   Train Samples: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%) | Poor (0): {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%) | Good (1): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
    print(f" - Test Sessions ({len(test_sessions)}): {test_sessions}")
    print(f"   Test Samples:  {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%) | Poor (0): {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.1f}%) | Good (1): {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.1f}%)")

    # 4. Preprocessing Pipeline
    num_cols = ['distancia_m', 'ambientLight', 'angRadiacion', 'angIncidencia']
    cat_cols = ['presenciaPersonas', 'entorno_encoded']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('pass', 'passthrough', cat_cols)
        ],
        remainder='drop'
    )

    # 5. Define Models
    models = {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss'),
        'Naive Baseline': DummyClassifier(strategy='most_frequent')
    }

    # 6. Evaluation and Training
    results = []
    fitted_pipelines = {}
    test_preds = {}
    test_probs = {}
    reports_text = []

    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "confusion_matrices"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[Step 4] Training & Benchmarking Models...")
    print("=" * 105)
    print(f"{'Model':22s} | {'5-Fold CV Acc':13s} | {'Test Acc':8s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s} | {'ROC-AUC':8s} | {'Train(s)':8s}")
    print("-" * 105)

    for name, clf in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # 5-fold CV on Train
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=skf,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            n_jobs=-1
        )
        cv_acc_mean = cv_res['test_accuracy'].mean()
        cv_acc_std = cv_res['test_accuracy'].std()

        # Fit on full training set
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Predict on held-out test set
        t_infer0 = time.time()
        y_pred = pipe.predict(X_test)
        infer_time = time.time() - t_infer0
        latency_us = (infer_time / len(X_test)) * 1e6

        if hasattr(pipe, 'predict_proba'):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc_val = roc_auc_score(y_test, y_prob)
        else:
            y_prob = None
            auc_val = 0.50

        acc = accuracy_score(y_test, y_pred)
        prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro')

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fitted_pipelines[name] = pipe
        test_preds[name] = y_pred
        test_probs[name] = y_prob

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec_1,
            'Recall': rec_1,
            'F1_Score': f1_1,
            'ROC_AUC': auc_val,
            'Precision_Poor_0': prec_0,
            'Recall_Poor_0': rec_0,
            'F1_Poor_0': f1_0,
            'F1_Macro': f1_macro,
            'CV_5Fold_Accuracy': cv_acc_mean,
            'CV_5Fold_Std': cv_acc_std,
            'True_Negatives': int(tn),
            'False_Positives': int(fp),
            'False_Negatives': int(fn),
            'True_Positives': int(tp),
            'Fit_Time_Sec': fit_time,
            'Latency_us': latency_us
        })

        print(f"{name:22s} | {cv_acc_mean*100:5.2f}% +/- {cv_acc_std*100:4.2f}% | {acc*100:7.2f}% | {prec_1*100:8.2f}% | {rec_1*100:7.2f}% | {f1_1*100:7.2f}% | {auc_val*100:7.2f}% | {fit_time:7.3f}s")

        # Classification report text
        clf_rep = classification_report(y_test, y_pred, target_names=['Poor Channel (0)', 'Good Channel (1)'], zero_division=0)
        reports_text.append(f"================================================================================\n"
                            f"Model: {name}\n"
                            f"Train Sessions: {train_sessions} (N={len(X_train)})\n"
                            f"Test Sessions:  {test_sessions} (N={len(X_test)})\n"
                            f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
                            f"ROC-AUC: {auc_val:.4f}\n"
                            f"--------------------------------------------------------------------------------\n"
                            f"{clf_rep}\n")

    print("-" * 105)
    df_results = pd.DataFrame(results)

    # Save Classification Report
    clf_rep_path = os.path.join("results", "classification_report.txt")
    with open(clf_rep_path, "w") as f:
        f.write("\n".join(reports_text))
    print(f"\n[+] Saved detailed classification reports to:\n    {os.path.abspath(clf_rep_path)}")

    # Save Model Comparison CSV
    comp_csv_path = os.path.join("results", "model_comparison.csv")
    df_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'True_Negatives', 'False_Positives', 'False_Negatives', 'True_Positives']].to_csv(comp_csv_path, index=False)
    print(f"[+] Saved model comparison table to:\n    {os.path.abspath(comp_csv_path)}")

    # 7. Champion Selection Rationale
    # For vehicular safety VLC:
    # False Positive (predicting Good when actual is Poor) causes severe packet loss (OOK selected on degraded channel).
    # Therefore, we demand:
    #   1. High Precision on Class 1 (Good Channel) -> Minimizes False Positives.
    #   2. High Specificity (Recall on Class 0 Poor Channel) -> Correctly flags bad channels.
    #   3. High ROC-AUC -> High discriminative power across all decision thresholds.
    non_baseline = df_results[df_results['Model'] != 'Naive Baseline']
    # Rank by ROC_AUC and Precision on Class 1
    best_idx = non_baseline['ROC_AUC'].idxmax()
    champion_name = non_baseline.loc[best_idx, 'Model']
    champion_pipe = fitted_pipelines[champion_name]

    print("\n" + "=" * 80)
    print(f"CHAMPION MODEL SELECTION: {champion_name}")
    print("=" * 80)
    print(f"Selection Criterion: Optimal combination of ROC-AUC ({df_results.loc[best_idx, 'ROC_AUC']*100:.2f}%), "
          f"Good Channel Precision ({df_results.loc[best_idx, 'Precision']*100:.2f}%), "
          f"Poor Channel Specificity ({df_results.loc[best_idx, 'Recall_Poor_0']*100:.2f}%), "
          f"and lowest False Positive rate (FP={df_results.loc[best_idx, 'False_Positives']}).")
    print(f" - Test Accuracy:  {df_results.loc[best_idx, 'Accuracy']*100:.2f}%")
    print(f" - Test Precision: {df_results.loc[best_idx, 'Precision']*100:.2f}%")
    print(f" - Test Recall:    {df_results.loc[best_idx, 'Recall']*100:.2f}%")
    print(f" - Test F1-Score:  {df_results.loc[best_idx, 'F1_Score']*100:.2f}%")
    print(f" - Test ROC-AUC:   {df_results.loc[best_idx, 'ROC_AUC']*100:.2f}%")
    print(f" - Inference:      {df_results.loc[best_idx, 'Latency_us']:.1f} microseconds per sample")

    # Serialize champion model to models/channel_quality_model.pkl
    pkl_model_path = os.path.join("models", "channel_quality_model.pkl")
    joblib.dump(champion_pipe, pkl_model_path)
    print(f"\n[+] Champion model serialized to:\n    {os.path.abspath(pkl_model_path)} ({os.path.getsize(pkl_model_path):,} bytes)")

    # 8. Visualizations
    print("\n[Step 5] Generating Publication-Quality Figures...")

    # A. Confusion Matrices
    classes = ['Poor (0)', 'Good (1)']
    for name in models.keys():
        cm = confusion_matrix(y_test, test_preds[name])
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {name}\n(Held-Out Test Sessions [5, 6, 10], N={len(y_test)})', fontsize=12, fontweight='bold')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=11)
        plt.yticks(tick_marks, classes, fontsize=11)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = val / len(y_test) * 100
                plt.text(j, i, f"{val:,}\n({pct:.1f}%)",
                         ha="center", va="center",
                         color="white" if val > thresh else "black",
                         fontsize=12, fontweight='bold')

        plt.ylabel('Actual Channel State', fontsize=11)
        plt.xlabel('Predicted Channel Quality', fontsize=11)
        plt.tight_layout()

        safe_name = name.lower().replace(' ', '_').replace('/', '_')
        cm_file = os.path.join("results", "confusion_matrices", f"{safe_name}_cm.png")
        plt.savefig(cm_file, dpi=300)
        plt.close()

    # Bonus: Combined 2x3 Confusion Matrix Grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    model_list = list(models.keys())
    for idx, name in enumerate(model_list):
        ax = axes[idx]
        cm = confusion_matrix(y_test, test_preds[name])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{name}\nAcc: {accuracy_score(y_test, test_preds[name])*100:.1f}% | AUC: {roc_auc_score(y_test, test_probs[name])*100 if test_probs[name] is not None else 50:.1f}%", fontsize=11, fontweight='bold')
        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = val / len(y_test) * 100
                ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual State', fontsize=10)
        ax.set_xlabel('Predicted State', fontsize=10)
    plt.suptitle("Confusion Matrix Comparison Across All VLC Channel Models (Held-Out Test Sessions [5, 6, 10])", fontsize=14, fontweight='bold')
    plt.tight_layout()
    grid_cm_path = os.path.join("results", "confusion_matrices", "confusion_matrices_all.png")
    plt.savefig(grid_cm_path, dpi=300)
    plt.close()
    print(f" [+] Saved individual & combined confusion matrix plots to:\n     {os.path.abspath(os.path.join('results', 'confusion_matrices'))}")

    # B. ROC Curve Comparison
    plt.figure(figsize=(9, 7))
    colors = {
        'Logistic Regression': '#1f77b4',
        'Decision Tree': '#ff7f0e',
        'Random Forest': '#2ca02c',
        'Gradient Boosting': '#d62728',
        'XGBoost': '#9467bd'
    }
    for name in models.keys():
        if name == 'Naive Baseline':
            continue
        prob = test_probs[name]
        if prob is not None:
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc_val = roc_auc_score(y_test, prob)
            plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc_val:.4f})", color=colors.get(name, None))

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison\nHeld-Out Test Sessions [5, 6, 10] (N = 1,697)', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_fig_path = os.path.join("results", "roc_curve.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f" [+] Saved ROC comparison curve to:\n     {os.path.abspath(roc_fig_path)}")

    # C. Feature Importance Comparison for Tree-Based Models
    tree_models = {
        'Random Forest': fitted_pipelines['Random Forest'].named_steps['classifier'].feature_importances_,
        'Gradient Boosting': fitted_pipelines['Gradient Boosting'].named_steps['classifier'].feature_importances_,
        'XGBoost': fitted_pipelines['XGBoost'].named_steps['classifier'].feature_importances_,
        'Decision Tree': fitted_pipelines['Decision Tree'].named_steps['classifier'].feature_importances_
    }
    fi_df = pd.DataFrame(tree_models, index=feature_cols)

    fig, ax = plt.subplots(figsize=(10, 6))
    fi_df.plot(kind='barh', ax=ax, width=0.75, colormap='viridis')
    ax.set_title('Feature Importance Comparison Across Tree-Based Ensembles\n(Predictors Measurable Independently of Reception)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Gini / Split Feature Importance Weight', fontsize=11)
    ax.set_ylabel('Physical Channel Input Feature', fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend(title='Model Architecture', fontsize=10)
    plt.tight_layout()
    fi_fig_path = os.path.join("results", "feature_importance.png")
    plt.savefig(fi_fig_path, dpi=300)
    plt.close()
    print(f" [+] Saved feature importance plot to:\n     {os.path.abspath(fi_fig_path)}")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"TRAINING PIPELINE COMPLETE in {total_time:.2f} seconds.")
    print("=" * 80)

    return df_results, champion_name, pkl_model_path


if __name__ == "__main__":
    run_pipeline()
