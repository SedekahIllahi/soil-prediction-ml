import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ml.schema import MODEL_FEATURES
from ml.adapters.adapter_registry import get_adapter
from ml.pipeline.splitting import DataSplitter
from ml.pipeline.preprocessing import build_linear_preprocessor, build_tree_preprocessor
from ml.pipeline.target_encoding import TargetEncoderWrapper
from ml.models.registry import get_baseline_models
from ml.pipeline.training import run_baseline_training
from ml.pipeline.evaluation import evaluate_model

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    # Load environment variables
    load_dotenv()
    
    dataset_path = os.environ.get("DATASET_PATH", "storage/datasets/urban_road_collapse_risk_dataset.csv")
    output_dir = Path("experiments/baseline")
    ensure_dir(output_dir)
    ensure_dir(output_dir / "confusion_matrices")
    
    random_seed = int(os.environ.get("RANDOM_SEED", 42))
    
    logger.info(f"Starting Phase 1 Baseline Training")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return
        
    # 1. Load and Validate
    logger.info("Loading and validating dataset...")
    adapter = get_adapter("urban_road_collapse")
    raw_df = adapter.load(dataset_path)
    
    report = adapter.validate(raw_df)
    logger.info(report.summary())
    
    if not report.is_valid:
        logger.error("Dataset validation failed. Aborting training.")
        return
        
    X_full, y_full, _ = adapter.transform(raw_df)
    
    # 2. Split
    logger.info("Splitting dataset...")
    splitter = DataSplitter()
    split_res = splitter.split(X_full, y_full)
    logger.info(f"Split sizes -> Train: {split_res.train_size}, Val: {split_res.val_size}, Test: {split_res.test_size}")
    
    # 3. Build Preprocessors
    logger.info("Building preprocessors...")
    linear_preprocessor = build_linear_preprocessor(MODEL_FEATURES)
    tree_preprocessor = build_tree_preprocessor(MODEL_FEATURES)
    
    target_encoder_linear = TargetEncoderWrapper(is_linear=True)
    target_encoder_tree = TargetEncoderWrapper(is_linear=False)
    
    # 4. Get Models
    models_to_train = get_baseline_models(random_seed=random_seed)
    logger.info(f"Models to train: {[m.name for m in models_to_train]}")
    
    # 5. Train
    logger.info("Starting training...")
    trained_models, fitted_lp, fitted_tp, fitted_tel, fitted_tet = run_baseline_training(
        split_res.X_train,
        split_res.y_train,
        linear_preprocessor,
        tree_preprocessor,
        target_encoder_linear,
        target_encoder_tree,
        models_to_train
    )
    
    # 6. Evaluate
    logger.info("Evaluating models on validation set...")
    all_metrics = []
    
    for tm in trained_models:
        # Select correct preprocessor for evaluation
        if tm.config.model_family == "linear":
            prep = fitted_lp
            enc = fitted_tel
        else:
            prep = fitted_tp
            enc = fitted_tet
            
        eval_res = evaluate_model(tm, split_res.X_val, split_res.y_val, prep, enc)
        all_metrics.append(eval_res.to_dict())
        
        # Save confusion matrix specifically
        cm_path = output_dir / "confusion_matrices" / f"{tm.config.name}_cm.json"
        with open(cm_path, "w") as f:
            json.dump({
                "model": tm.config.name,
                "labels": eval_res.class_labels,
                "matrix": eval_res.confusion_matrix
            }, f, indent=2)
            
        logger.info(f"Model {tm.config.name}: Weighted F1 = {eval_res.weighted_f1:.4f}, Accuracy = {eval_res.accuracy:.4f} (Train time: {tm.training_time_seconds:.2f}s)")
        
    # 7. Save Metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    logger.info(f"Baseline training complete. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
