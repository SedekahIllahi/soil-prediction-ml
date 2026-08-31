"""
Multi-model comparison and ranking.
Ranks trained models by weighted F1-Score (primary) and
High Risk recall (secondary) as specified in ARCHITECTURE.md §6.
"""
from dataclasses import dataclass, field
from ml.pipeline.evaluation import EvaluationResult

@dataclass
class ComparisonEntry:
    """A single model's ranking entry in the comparison table."""
    rank: int
    model_name: str
    weighted_f1: float
    macro_f1: float
    accuracy: float
    high_class_recall: float
    training_time_seconds: float
    evaluation_result: EvaluationResult

    @property
    def high_risk_recall(self) -> float:
        """Backward compatibility alias for high_class_recall."""
        return self.high_class_recall

@dataclass
class ComparisonReport:
    """Full comparison report across all candidate models."""
    entries: list[ComparisonEntry] = field(default_factory=list)
    best_model_name: str = ""
    primary_metric: str = "weighted_f1"
    secondary_metric: str = "high_class_recall"

    def to_dict(self) -> dict:
        return {
            "best_model_name": self.best_model_name,
            "primary_metric": self.primary_metric,
            "secondary_metric": self.secondary_metric,
            "rankings": [
                {
                    "rank": entry.rank,
                    "model_name": entry.model_name,
                    "weighted_f1": round(entry.weighted_f1, 4),
                    "macro_f1": round(entry.macro_f1, 4),
                    "accuracy": round(entry.accuracy, 4),
                    "high_class_recall": round(entry.high_class_recall, 4),
                    "high_risk_recall": round(entry.high_class_recall, 4),  # Alias for backward compatibility
                    "training_time_seconds": round(entry.training_time_seconds, 3),
                }
                for entry in self.entries
            ],
        }


def compare_models(
    evaluation_results: list[EvaluationResult],
    training_times: dict[str, float],
) -> ComparisonReport:
    """
    Compares evaluated models and ranks them by weighted F1-Score (primary)
    with High Risk recall as a tiebreaker (secondary).

    Args:
        evaluation_results: List of EvaluationResult objects from evaluation.
        training_times: Dict mapping model_name -> training_time_seconds.

    Returns:
        ComparisonReport with ranked entries and the best model identified.
    """
    if not evaluation_results:
        return ComparisonReport()

    entries: list[ComparisonEntry] = []
    for result in evaluation_results:
        # Extract High-class recall specifically; default to 0 if class not present
        high_class_recall = result.per_class.get("High", {}).get("recall", 0.0)

        entries.append(
            ComparisonEntry(
                rank=0,  # assigned after sorting
                model_name=result.model_name,
                weighted_f1=result.weighted_f1,
                macro_f1=result.macro_f1,
                accuracy=result.accuracy,
                high_class_recall=high_class_recall,
                training_time_seconds=training_times.get(result.model_name, 0.0),
                evaluation_result=result,
            )
        )

    # Sort by weighted F1 descending, then High-class recall descending as tiebreaker
    entries.sort(key=lambda e: (e.weighted_f1, e.high_class_recall), reverse=True)

    # Assign ranks (1-indexed)
    for i, entry in enumerate(entries):
        entry.rank = i + 1

    return ComparisonReport(
        entries=entries,
        best_model_name=entries[0].model_name,
    )
