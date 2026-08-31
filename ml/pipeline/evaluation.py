from dataclasses import dataclass, asdict
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

from ml.pipeline.training import TrainedModel
from ml.schema import TARGET_CLASSES

@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]  # {class_name: {precision, recall, f1}}
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    
    def to_dict(self) -> dict:
        return asdict(self)

def evaluate_model(
    trained_model: TrainedModel, 
    X_val_raw: pd.DataFrame, 
    y_val_raw: pd.Series, 
    preprocessor,
    target_encoder
) -> EvaluationResult:
    """
    Evaluates a trained model on validation data.
    
    Args:
        trained_model: The TrainedModel to evaluate.
        X_val_raw: Unprocessed validation features.
        y_val_raw: Unprocessed validation target (strings).
        preprocessor: Fitted preprocessor appropriate for the model family.
        target_encoder: Fitted TargetEncoderWrapper appropriate for the model family.
        
    Returns:
        EvaluationResult containing metrics and confusion matrix.
    """
    # 1. Transform features only using fitted preprocessor
    X_val = preprocessor.transform(X_val_raw)
    
    # 2. Predict encoded class indices/values
    y_pred_encoded = trained_model.model.predict(X_val)
    
    # 3. Decode predictions back to canonical string labels for evaluation
    y_pred_decoded = target_encoder.decode(y_pred_encoded)
    
    # 4. Calculate metrics (using the string labels)
    labels = list(TARGET_CLASSES)
    
    accuracy = accuracy_score(y_val_raw, y_pred_decoded)
    
    # Macro metrics
    macro_precision = precision_score(y_val_raw, y_pred_decoded, labels=labels, average='macro', zero_division=0)
    macro_recall = recall_score(y_val_raw, y_pred_decoded, labels=labels, average='macro', zero_division=0)
    macro_f1 = f1_score(y_val_raw, y_pred_decoded, labels=labels, average='macro', zero_division=0)
    
    # Weighted metric (primary comparison metric)
    weighted_f1 = f1_score(y_val_raw, y_pred_decoded, labels=labels, average='weighted', zero_division=0)
    
    # Per-class metrics
    class_precision = precision_score(y_val_raw, y_pred_decoded, labels=labels, average=None, zero_division=0)
    class_recall = recall_score(y_val_raw, y_pred_decoded, labels=labels, average=None, zero_division=0)
    class_f1 = f1_score(y_val_raw, y_pred_decoded, labels=labels, average=None, zero_division=0)
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(class_precision[i]),
            "recall": float(class_recall[i]),
            "f1": float(class_f1[i])
        }
        
    # Confusion matrix
    cm = confusion_matrix(y_val_raw, y_pred_decoded, labels=labels)
    cm_list = cm.tolist()
    
    return EvaluationResult(
        model_name=trained_model.config.name,
        accuracy=float(accuracy),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class=per_class,
        confusion_matrix=cm_list,
        class_labels=labels
    )
