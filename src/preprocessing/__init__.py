"""
Subpacote de Pré-processamento e Carga de Dados (src/preprocessing)
"""

from src.preprocessing.data_loader import load_gold_silver_data
from src.preprocessing.pipeline import (
    EducationFeatureEngineer,
    build_preprocessor_pipeline,
    build_feature_dictionary,
    extract_transformed_feature_names,
)

__all__ = [
    "load_gold_silver_data",
    "EducationFeatureEngineer",
    "build_preprocessor_pipeline",
    "build_feature_dictionary",
    "extract_transformed_feature_names",
]

