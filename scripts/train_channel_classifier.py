#!/usr/bin/env python3
"""
scripts/train_channel_classifier.py

Academic Project:
AI-BASED ADAPTIVE CAR-TO-CAR VISIBLE LIGHT COMMUNICATION WITH ML-BASED CHANNEL SENSING

Step 3: Reproducible ML Training & Benchmarking Pipeline for Binary Channel-Quality Classification
- Target: target_channel_quality_binary (1 = Good Channel [PL >= -35.0 dB], 0 = Poor Channel [PL < -35.0 dB])
- Inputs: Strictly vehicle-available physical inputs:
          distancia_m, ambientLight, angRadiacion, angIncidencia, presenciaPersonas, entorno_encoded
- Zero target leakage: 0 received-signal variables used.
- Reproducibility: Fixed seeds (random_state=42), self-contained pipeline, automated artifact persistence.
- Models evaluated:
    0. Dummy (Naive Baseline)
    1. Logistic Regression (L2 Regularized)
    2. Decision Tree (Interpretable baseline)
    3. Random Forest Classifier
    4. Gradient Boosting Classifier
    5. HistGradientBoosting Classifier
    6. K-Nearest Neighbors Classifier
    7. Multi-Layer Perceptron (MLP Neural Network)
- Evaluation:
    * 80/20 Stratified Train-Test Split
    * 5-Fold Stratified Cross-Validation on Train
    * 5-Fold GroupKFold Cross-Validation on Train (grouped by session_id)
    * Comprehensive metrics: Accuracy, Balanced Acc, Precision, Recall, F1, ROC-AUC, PR-AUC, Latency
    * Publication-quality visual plots saved to reports/
    * Best pipeline serialized to models/best_vlc_binary_classifier.joblib
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance
import joblib

# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def run_training_pipeline():
    print("=" * 80)
    print("AI-BASED ADAPTIVE CAR-TO-CAR VLC: BINARY CHANNEL CLASSIFICATION PIPELINE")
    print("=" * 80)
    start_total_time = time.time()

    # 1. Load ML-Ready Dataset (Read-Only)
    dataset_path = os.path.join("dataset", "ml_ready", "vlc_channel_ml_ready.csv")
    print(f"\n[Phase 1] Loading verified ML-ready dataset from:\n  {os.path.abspath(dataset_path)}")
    df = pd.read_csv(dataset_path)
    n_samples, n_cols = df.shape
    print(f"Loaded {n_samples:,} samples across {n_cols} columns.")

    # 2. Define Features and Target
    target_col = 'target_channel_quality_binary'
    metadata_col = 'session_id'
    feature_cols = [
        'distancia_m',
        'ambientLight',
        'angRadiacion',
        'angIncidencia',
        'presenciaPersonas',
        'entorno_encoded'
    ]

    print(f"\n[Phase 2] Pipeline Configuration:")
    print(f" - Target: '{target_col}' (Binary: 0 = Poor, 1 = Good)")
    print(f" - Physical Input Features ({len(feature_cols)}): {feature_cols}")
    print(f" - Group Metadata: '{metadata_col}' (12 experimental sessions)")

    # Assert zero leakage
    prohibited = ['pathloss', 'vpp', 'p_rx', 'snr', 'ber', 'p_amb', 'outage']
    for feat in feature_cols:
        for p in prohibited:
            assert p not in feat.lower(), f"CRITICAL LEAKAGE: {feat}"
    print(" [+] Target leakage audit: ZERO prohibited received-signal variables in feature set.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df[metadata_col].copy()

    # 3. Train-Test Split (80% Train, 20% Held-Out Test)
    # Stratified to ensure exact class balance preservation
    print("\n[Phase 3] Splitting Dataset into Train (80%) and Held-Out Test (20%)...")
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.20, random_state=42, stratify=y
    )
    print(f" - Training Set: {len(X_train):,} samples (Class 0: {(y_train == 0).sum():,}, Class 1: {(y_train == 1).sum():,})")
    print(f" - Held-Out Test Set: {len(X_test):,} samples (Class 0: {(y_test == 0).sum():,}, Class 1: {(y_test == 1).sum():,})")

    # 4. Strict Preprocessing Pipeline
    # Continuous features scaled with StandardScaler; discrete/binary features passed through
    continuous_features = ['distancia_m', 'ambientLight', 'angRadiacion', 'angIncidencia']
    binary_features = ['presenciaPersonas', 'entorno_encoded']

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), continuous_features),
            ('pass', 'passthrough', binary_features)
        ],
        remainder='drop'
    )

    # 5. Define Model Candidates
    print("\n[Phase 4] Instantiating Model Benchmarking Suite...")
    models = {
        'Naive Baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=6, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Multi-Layer Perceptron': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    # Cross-validation strategies
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)

    results = []
    fitted_pipelines = {}
    test_predictions = {}
    test_probabilities = {}

    print("\n" + "=" * 80)
    print(f"{'Model Name':24s} | {'CV Acc':7s} | {'Grp-CV':7s} | {'Test Acc':8s} | {'Bal Acc':7s} | {'F1-Mac':7s} | {'ROC-AUC':7s} | {'Train(s)':8s}")
    print("-" * 80)

    for name, clf in models.items():
        # Build complete pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        t0 = time.time()
        # Stratified 5-Fold CV on training set
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=skf,
            scoring=['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc'],
            n_jobs=-1
        )
        cv_acc_mean = cv_scores['test_accuracy'].mean()
        cv_acc_std = cv_scores['test_accuracy'].std()
        cv_bal_acc_mean = cv_scores['test_balanced_accuracy'].mean()
        cv_f1_mean = cv_scores['test_f1_macro'].mean()
        cv_auc_mean = cv_scores['test_roc_auc'].mean()

        # Group 5-Fold CV on training set (by session_id)
        # Tests generalization across unseen experimental runs
        grp_cv_scores = cross_validate(
            pipe, X_train, y_train, groups=groups_train, cv=gkf,
            scoring=['accuracy', 'balanced_accuracy', 'f1_macro'],
            n_jobs=-1
        )
        grp_cv_acc_mean = grp_cv_scores['test_accuracy'].mean()
        grp_cv_acc_std = grp_cv_scores['test_accuracy'].std()

        # Fit on full training set
        fit_t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_duration = time.time() - fit_t0

        # Predict on held-out test set
        infer_t0 = time.time()
        y_pred = pipe.predict(X_test)
        infer_duration = time.time() - infer_t0
        latency_us = (infer_duration / len(X_test)) * 1e6  # Microseconds per sample

        # Predict probabilities if supported
        if hasattr(pipe, 'predict_proba'):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob)
            test_pr_auc = average_precision_score(y_test, y_prob)
        else:
            y_prob = None
            test_auc = 0.5
            test_pr_auc = 0.5

        test_acc = accuracy_score(y_test, y_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        test_prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        test_prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        test_rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        test_rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        test_f1_macro = f1_score(y_test, y_pred, average='macro')
        test_f1_weighted = f1_score(y_test, y_pred, average='weighted')

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fitted_pipelines[name] = pipe
        test_predictions[name] = y_pred
        test_probabilities[name] = y_prob

        results.append({
            'Model': name,
            'CV_Accuracy_Mean': cv_acc_mean,
            'CV_Accuracy_Std': cv_acc_std,
            'GroupCV_Accuracy_Mean': grp_cv_acc_mean,
            'GroupCV_Accuracy_Std': grp_cv_acc_std,
            'Test_Accuracy': test_acc,
            'Test_Balanced_Accuracy': test_bal_acc,
            'Test_Precision_Poor_0': test_prec_0,
            'Test_Precision_Good_1': test_prec_1,
            'Test_Recall_Poor_0': test_rec_0,
            'Test_Recall_Good_1': test_rec_1,
            'Test_F1_Macro': test_f1_macro,
            'Test_F1_Weighted': test_f1_weighted,
            'Test_ROC_AUC': test_auc,
            'Test_PR_AUC': test_pr_auc,
            'True_Negatives_0': int(tn),
            'False_Positives_0': int(fp),
            'False_Negatives_1': int(fn),
            'True_Positives_1': int(tp),
            'Fit_Time_Seconds': fit_duration,
            'Latency_us_per_sample': latency_us
        })

        print(f"{name:24s} | {cv_acc_mean*100:5.2f}% | {grp_cv_acc_mean*100:5.2f}% | {test_acc*100:6.2f}% | {test_bal_acc*100:5.2f}% | {test_f1_macro*100:5.2f}% | {test_auc*100:5.2f}% | {fit_duration:6.3f}s")

    print("-" * 80)
    df_results = pd.DataFrame(results)

    # 6. Champion Model Selection
    # Highest Test F1-Macro and Test ROC-AUC among non-naive models
    non_baseline = df_results[df_results['Model'] != 'Naive Baseline']
    best_idx = non_baseline['Test_F1_Macro'].idxmax()
    best_model_name = df_results.loc[best_idx, 'Model']
    best_pipeline = fitted_pipelines[best_model_name]

    print(f"\n[Phase 5] CHAMPION MODEL SELECTED: **{best_model_name}**")
    print(f" - Test Accuracy:          {df_results.loc[best_idx, 'Test_Accuracy']*100:.2f}%")
    print(f" - Test Balanced Accuracy: {df_results.loc[best_idx, 'Test_Balanced_Accuracy']*100:.2f}%")
    print(f" - Test F1-Score (Macro):  {df_results.loc[best_idx, 'Test_F1_Macro']*100:.2f}%")
    print(f" - Test ROC-AUC:           {df_results.loc[best_idx, 'Test_ROC_AUC']*100:.2f}%")
    print(f" - Inference Latency:      {df_results.loc[best_idx, 'Latency_us_per_sample']:.1f} microseconds/sample")

    # 7. Detailed Feature Importance Analysis on Champion Model
    print(f"\n[Phase 6] Computing Feature Importance for {best_model_name}...")
    perm_importance = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        'Feature': feature_cols,
        'Permutation_Importance_Mean': perm_importance.importances_mean,
        'Permutation_Importance_Std': perm_importance.importances_std
    }).sort_values(by='Permutation_Importance_Mean', ascending=False)
    print(perm_df.to_string(index=False))

    # 8. Save Metrics & Serialize Best Pipeline
    print("\n[Phase 7] Exporting Models and Evaluation Metrics...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Export serialized pipeline
    model_export_path = os.path.join("models", "best_vlc_binary_classifier.joblib")
    joblib.dump(best_pipeline, model_export_path)
    print(f" [+] Serialized champion model saved to: {os.path.abspath(model_export_path)} ({os.path.getsize(model_export_path):,} bytes)")

    # Save metrics JSON and CSV
    metrics_json_path = os.path.join("models", "model_comparison_metrics.json")
    metrics_csv_path = os.path.join("models", "model_comparison_metrics.csv")
    df_results.to_csv(metrics_csv_path, index=False)
    with open(metrics_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f" [+] Metrics table saved to:             {os.path.abspath(metrics_csv_path)}")

    # 9. Generate Publication-Quality Visualizations
    print("\n[Phase 8] Rendering Performance Plots...")

    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(df_results))
    width = 0.22
    plt.bar(x_pos - width, df_results['CV_Accuracy_Mean'] * 100, width, label='5-Fold CV Accuracy (%)', color='#4A90E2')
    plt.bar(x_pos, df_results['Test_Accuracy'] * 100, width, label='Held-Out Test Accuracy (%)', color='#50E3C2')
    plt.bar(x_pos + width, df_results['Test_F1_Macro'] * 100, width, label='Test F1-Macro (%)', color='#F5A623')
    plt.xticks(x_pos, df_results['Model'], rotation=25, ha='right', fontsize=10)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Vehicle VLC Channel Quality: Binary Classification Benchmark Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(40, 100)
    plt.tight_layout()
    comp_plot_path = os.path.join("reports", "model_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f" [+] Benchmark Comparison plot:         {os.path.abspath(comp_plot_path)}")

    # Plot 2: ROC Curves
    plt.figure(figsize=(9, 7))
    for name in models.keys():
        if name == 'Naive Baseline':
            continue
        prob = test_probabilities[name]
        if prob is not None:
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc_val = roc_auc_score(y_test, prob)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves on Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_plot_path = os.path.join("reports", "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f" [+] ROC Curves plot:                   {os.path.abspath(roc_plot_path)}")

    # Plot 3: Confusion Matrix of Champion Model
    cm = confusion_matrix(y_test, test_predictions[best_model_name])
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {best_model_name}\n(Held-Out Test Set, N = {len(y_test)})', fontsize=13, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ['Poor Channel (0)', 'Good Channel (1)']
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / len(y_test)) * 100
            plt.text(j, i, f"{val:,}\n({pct:.1f}%)",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=12, fontweight='bold')

    plt.ylabel('Ground-Truth Class (Physical Channel State)', fontsize=12)
    plt.xlabel('Model Predicted Class', fontsize=12)
    plt.tight_layout()
    cm_plot_path = os.path.join("reports", "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f" [+] Confusion Matrix plot:             {os.path.abspath(cm_plot_path)}")

    # Plot 4: Permutation Feature Importance
    plt.figure(figsize=(9, 5))
    perm_sorted = perm_df.sort_values(by='Permutation_Importance_Mean', ascending=True)
    plt.barh(perm_sorted['Feature'], perm_sorted['Permutation_Importance_Mean'] * 100, xerr=perm_sorted['Permutation_Importance_Std'] * 100, color='#2ECC71', alpha=0.85, capsize=4)
    plt.xlabel('Mean Decrease in Accuracy (% drop when shuffled)', fontsize=12)
    plt.title(f'Permutation Feature Importance on Held-Out Test Set ({best_model_name})', fontsize=13, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fi_plot_path = os.path.join("reports", "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f" [+] Feature Importance plot:           {os.path.abspath(fi_plot_path)}")

    total_pipeline_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETE in {total_pipeline_time:.2f} seconds.")
    print("=" * 80)

    return df_results, best_model_name, best_pipeline


if __name__ == "__main__":
    run_training_pipeline()
