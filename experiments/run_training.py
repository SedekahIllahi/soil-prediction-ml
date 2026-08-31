"""
Standalone Baseline Training & Evaluation Runner.
Executes Phase 3 training across all candidate algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM (RBF kernel)

Evaluates on Validation set (15%) for model selection and Test set (15%) for final verification.
Saves artifacts and outputs comprehensive evaluation matrices & report.
"""
import sys
import os
import time
import json
import argparse
import pandas as pd
import numpy as np

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.adapters.urban_road_collapse import UrbanRoadCollapseAdapter
from ml.models.registry import get_baseline_models
from ml.pipeline.splitting import DataSplitter
from ml.pipeline.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
    save_preprocessor,
)
from ml.pipeline.target_encoding import TargetEncoderWrapper
from ml.pipeline.training import run_baseline_training
from ml.pipeline.evaluation import evaluate_model
from ml.pipeline.comparison import compare_models
from ml.schema import MODEL_FEATURES, TARGET_CLASSES

def main():
    parser = argparse.ArgumentParser(description="Run Soil Risk ML Baseline Model Training & Evaluation")
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "storage", "datasets", "urban_road_collapse_risk_dataset.csv"),
        help="Path to CSV dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "experiments"),
        help="Output directory for reports and metrics"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("==========================================================================")
    print("        SOIL RISK ML PREDICTION SYSTEM — PHASE 3 MODEL TRAINING           ")
    print("==========================================================================")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found at {args.dataset_path}")
        sys.exit(1)

    # 1. Ingest dataset
    print("\n[1/5] Ingesting and adapting dataset...")
    raw_df = pd.read_csv(args.dataset_path)
    print(f"Loaded raw dataset shape: {raw_df.shape}")

    adapter = UrbanRoadCollapseAdapter()
    X, y, metadata = adapter.transform(raw_df)
    print(f"Extracted feature matrix X: {X.shape}, target y: {y.shape[0]} rows")

    # 2. Stratified 70/15/15 Split
    print("\n[2/5] Performing stratified 70/15/15 split...")
    splitter = DataSplitter(test_size=0.15, val_size=0.15)
    split = splitter.split(X, y)

    print(f"  Training set   (70%): {split.train_size} samples")
    print(f"  Validation set (15%): {split.val_size} samples")
    print(f"  Test set       (15%): {split.test_size} samples")

    # Target class distribution in train
    print("\nTrain class distribution:")
    print(split.y_train.value_counts())

    # 3. Initialize Preprocessors, Target Encoders, & Baseline Model Configs
    print("\n[3/5] Initializing preprocessing pipelines & candidate algorithms...")
    linear_preproc = build_linear_preprocessor(MODEL_FEATURES)
    tree_preproc = build_tree_preprocessor(MODEL_FEATURES)

    target_enc_linear = TargetEncoderWrapper(is_linear=True)
    target_enc_tree = TargetEncoderWrapper(is_linear=False)

    model_configs = get_baseline_models()
    print(f"Configured {len(model_configs)} baseline algorithms:")
    for cfg in model_configs:
        print(f"  - {cfg.name} (family: {cfg.model_family})")

    # 4. Train Models
    print("\n[4/5] Training baseline models on training split...")
    start_train_all = time.time()
    (
        trained_models,
        fitted_linear_preproc,
        fitted_tree_preproc,
        fitted_target_enc_linear,
        fitted_target_enc_tree,
    ) = run_baseline_training(
        X_train=split.X_train,
        y_train=split.y_train,
        linear_preprocessor=linear_preproc,
        tree_preprocessor=tree_preproc,
        target_encoder_linear=target_enc_linear,
        target_encoder_tree=target_enc_tree,
        model_configs=model_configs,
    )
    total_train_time = time.time() - start_train_all
    print(f"Completed model training in {total_train_time:.2f} seconds.")

    # 5. Evaluate on Validation Set
    print("\n[5/5] Evaluating all candidate models on Validation Set (15%)...")
    eval_results = []
    training_times = {}

    for tm in trained_models:
        preproc = fitted_linear_preproc if tm.config.model_family == "linear" else fitted_tree_preproc
        enc = fitted_target_enc_linear if tm.config.model_family == "linear" else fitted_target_enc_tree

        eval_res = evaluate_model(
            trained_model=tm,
            X_val_raw=split.X_val,
            y_val_raw=split.y_val,
            preprocessor=preproc,
            target_encoder=enc
        )
        eval_results.append(eval_res)
        training_times[tm.config.name] = tm.training_time_seconds
        print(f"  -> {tm.config.name:<20} Weighted F1: {eval_res.weighted_f1:.4f} | Accuracy: {eval_res.accuracy:.4f} | Time: {tm.training_time_seconds:.3f}s")

    # Compare & Rank
    comparison_report = compare_models(eval_results, training_times)

    # Find winning model
    winning_model_name = comparison_report.best_model_name
    winning_tm = next(tm for tm in trained_models if tm.config.name == winning_model_name)
    winning_preproc = fitted_linear_preproc if winning_tm.config.model_family == "linear" else fitted_tree_preproc
    winning_enc = fitted_target_enc_linear if winning_tm.config.model_family == "linear" else fitted_target_enc_tree

    # Evaluate Winner on Isolated Test Set
    test_eval_res = evaluate_model(
        trained_model=winning_tm,
        X_val_raw=split.X_test,
        y_val_raw=split.y_test,
        preprocessor=winning_preproc,
        target_encoder=winning_enc
    )

    # Print Summary Report
    print("\n==========================================================================")
    print("                     BASELINE MODEL RANKING SUMMARY                       ")
    print("==========================================================================")
    print(f"{'Rank':<5} {'Model Name':<20} {'Weighted F1':<12} {'Macro F1':<10} {'Accuracy':<10} {'High-Class Rec':<15} {'Train Time (s)':<12}")
    print("-" * 88)
    for entry in comparison_report.entries:
        print(f"{entry.rank:<5} {entry.model_name:<20} {entry.weighted_f1:<12.4f} {entry.macro_f1:<10.4f} {entry.accuracy:<10.4f} {entry.high_class_recall:<15.4f} {entry.training_time_seconds:<12.3f}")

    print("\n==========================================================================")
    print(f"TOP VALIDATION BASELINE MODEL: {winning_model_name.upper()}")
    print("==========================================================================")
    print("Validation Metrics (Used for Model Comparison):")
    val_win_eval = next(e for e in eval_results if e.model_name == winning_model_name)
    print(f"  Weighted F1: {val_win_eval.weighted_f1:.4f}")
    print(f"  Accuracy:    {val_win_eval.accuracy:.4f}")
    print(f"  Macro Rec:   {val_win_eval.macro_recall:.4f}")
    print("Final Test Set Evaluation (Held-out 15% Test Split — Not Used for Selection):")
    print(f"  Weighted F1: {test_eval_res.weighted_f1:.4f}")
    print(f"  Accuracy:    {test_eval_res.accuracy:.4f}")
    print(f"  Macro Rec:   {test_eval_res.macro_recall:.4f}")

    # Build Complete JSON Report Object
    full_report = {
        "dataset": {
            "path": args.dataset_path,
            "total_rows": len(raw_df),
            "train_rows": split.train_size,
            "val_rows": split.val_size,
            "test_rows": split.test_size,
            "target_classes": list(TARGET_CLASSES)
        },
        "ranking": comparison_report.to_dict(),
        "validation_evaluations": {res.model_name: res.to_dict() for res in eval_results},
        "best_model_test_evaluation": test_eval_res.to_dict()
    }

    report_file = os.path.join(args.output_dir, "baseline_training_report.json")
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSaved complete json evaluation report to: {report_file}")

if __name__ == "__main__":
    main()
