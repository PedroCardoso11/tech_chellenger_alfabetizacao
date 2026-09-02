"""
Subpacote de Avaliação e Otimização de Métricas (src/evaluation)
"""

from src.evaluation.metrics import evaluate_all_models_on_test
from src.evaluation.threshold import optimize_decision_threshold

__all__ = [
    "evaluate_all_models_on_test",
    "optimize_decision_threshold",
]

