"""
Package evaluation — métriques d'évaluation du pipeline Text2SQL.
"""

from evaluation.metrics import (
    MetricsCalculator,
    EvaluationReport,
    SingleEvalResult,
    REFERENCE_DATASET,
    normalize_sql,
    exact_match,
    table_precision,
    table_recall,
    table_f1,
)

__all__ = [
    "MetricsCalculator",
    "EvaluationReport",
    "SingleEvalResult",
    "REFERENCE_DATASET",
    "normalize_sql",
    "exact_match",
    "table_precision",
    "table_recall",
    "table_f1",
]
