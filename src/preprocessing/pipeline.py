"""
Módulo de Engenharia de Atributos e Pipeline de Pré-processamento (src/preprocessing/pipeline.py)
Criação de features compostas educacionais, socioeconômicas e territoriais REAIS,
imputação, escalonamento e codificação categórica com garantia estrita de Zero Data Leakage.
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    BASE_NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    CATEGORICAL_NOMINAL_LOW_CARD,
    CATEGORICAL_ORDINAL,
    CATEGORICAL_HIGH_CARD,
)


class EducationFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado Scikit-Learn para criação de índices compostos
    e atributos educacionais / socioeconômicos derivados de domínio sobre dados REAIS.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. Razão de Desempenho Escola vs Estado (Pressão de Aprendizado Relativo)
        esc_nao_alfab = pd.to_numeric(X_out.get("escola_percentual_nao_alfabetizado", 0.0), errors="coerce").fillna(0.0)
        uf_abaixo_meta = pd.to_numeric(X_out.get("uf_percentual_municipios_abaixo_meta", 50.0), errors="coerce").fillna(50.0)
        X_out["razao_desempenho_escola_uf"] = np.round(esc_nao_alfab / (uf_abaixo_meta + 1.0), 4)

        # 2. Índice de Engajamento e Eficácia Escolar (Presença x Sucesso Agregado)
        presenca_esc = pd.to_numeric(X_out.get("escola_percentual_presenca", 80.0), errors="coerce").fillna(80.0)
        taxa_sucesso_esc = np.clip(1.0 - (esc_nao_alfab / 100.0), 0.0, 1.0)
        X_out["indice_engajamento_escola"] = np.round(presenca_esc * taxa_sucesso_esc, 3)

        # 3. Razão Beneficiários Bolsa Família por Porte Escolar (Pressão Socioeconômica Territorial)
        bf_benef = pd.to_numeric(X_out.get("total_beneficiarios", 1000.0), errors="coerce").fillna(1000.0)
        porte_esc = pd.to_numeric(X_out.get("escola_total_alunos", 50.0), errors="coerce").replace(0, 50.0).fillna(50.0)
        X_out["razao_beneficiarios_porte_escola"] = np.round(bf_benef / (porte_esc + 1.0), 3)

        return X_out


def build_feature_dictionary(df_cols: List[str]) -> Dict[str, List[str]]:
    """Gera o dicionário com os grupos de atributos."""
    return {
        "numeric": BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES,
        "nominal": [c for c in CATEGORICAL_NOMINAL_LOW_CARD if c in df_cols],
        "ordinal": [c for c in CATEGORICAL_ORDINAL if c in df_cols],
        "high_card": [c for c in CATEGORICAL_HIGH_CARD if c in df_cols],
    }


def build_preprocessor_pipeline(feature_dict: dict) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer do Scikit-Learn contendo imputação,
    escalonamento numérico e codificação categórica.
    """
    numeric_cols = feature_dict["numeric"]
    nominal_cols = feature_dict["nominal"]
    ordinal_cols = feature_dict["ordinal"]
    high_card_cols = feature_dict["high_card"]

    # Pipeline Numérico: Imputação por Mediana + RobustScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    # Pipeline Categórico Nominal: Imputação por Moda + OneHotEncoder
    nom_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    # Pipeline Categórico Ordinal: Imputação por Moda + OrdinalEncoder
    ordinal_categories = [
        ["Mais de 10 p.p. abaixo", "Entre 5 e 10 p.p. abaixo", "Ate 5 p.p. abaixo", "Meta atingida"]
    ]
    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", RobustScaler()),
    ])
    
    # Pipeline para Alta Cardinalidade (UF): OneHotEncoder
    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("nom", nom_pipeline, nominal_cols),
            ("ord", ord_pipeline, ordinal_cols),
            ("high_card", high_card_pipeline, high_card_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def extract_transformed_feature_names(fitted_preprocessor: ColumnTransformer, feature_dict: dict) -> List[str]:
    """Extrai os nomes finais de todas as colunas pós-transformação."""
    feature_names = []
    
    # Numéricas
    feature_names.extend(feature_dict["numeric"])
    
    # Nominais One-Hot
    try:
        nom_ohe = fitted_preprocessor.named_transformers_["nom"].named_steps["onehot"]
        nom_cols = nom_ohe.get_feature_names_out(feature_dict["nominal"])
        feature_names.extend(list(nom_cols))
    except Exception:
        pass
        
    # Ordinais
    feature_names.extend(feature_dict["ordinal"])
    
    # High Card (UF) One-Hot
    try:
        uf_ohe = fitted_preprocessor.named_transformers_["high_card"].named_steps["onehot"]
        uf_cols = uf_ohe.get_feature_names_out(feature_dict["high_card"])
        feature_names.extend(list(uf_cols))
    except Exception:
        pass
        
    return feature_names


def split_and_preprocess_data(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE):
    """
    Divide treino e teste e aplica o pré-processador isolado no treino (Zero Data Leakage).
    Retorna X_train_proc, X_test_proc, y_train, y_test, feature_names, preprocessor.
    """
    X_raw = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    feature_dict = build_feature_dictionary(list(X_raw.columns))
    fe = EducationFeatureEngineer()
    X_train_fe = fe.transform(X_train_raw)
    X_test_fe = fe.transform(X_test_raw)
    
    preprocessor = build_preprocessor_pipeline(feature_dict)
    X_train_proc = preprocessor.fit_transform(X_train_fe)
    X_test_proc = preprocessor.transform(X_test_fe)
    
    feature_names = extract_transformed_feature_names(preprocessor, feature_dict)
    
    return X_train_proc, X_test_proc, y_train, y_test, feature_names, preprocessor
