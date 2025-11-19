"""
Módulo de preprocesamiento para modelos de Machine Learning.
Realiza encoding de variables categóricas y normalización de features numéricas.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logger = logging.getLogger(__name__)


def separar_features_target(df, target_col='price'):
    """
    Separa features y target del dataset.
    
    Args:
        df (pd.DataFrame): DataFrame completo
        target_col (str): Nombre de la columna target
        
    Returns:
        tuple: (X, y) donde X son features e y es el target
    """
    if target_col not in df.columns:
        logger.error(f"Columna '{target_col}' no encontrada")
        return None, None
    
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    
    logger.info(f"Features separados: {X.shape}")
    logger.info(f"Target separado: {y.shape}")
    
    return X, y


def identificar_tipos_features(X):
    """
    Identifica columnas categóricas y numéricas.
    
    Args:
        X (pd.DataFrame): DataFrame de features
        
    Returns:
        tuple: (categorical_cols, numerical_cols)
    """
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Columnas categóricas: {len(categorical_cols)}")
    logger.info(f"Columnas numéricas: {len(numerical_cols)}")
    
    return categorical_cols, numerical_cols


def encodear_categoricas(X_train, X_test, categorical_cols):
    """
    Codifica variables categóricas usando LabelEncoder.
    Fit SOLO con train, transform en train y test.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        X_test (pd.DataFrame): Features de prueba
        categorical_cols (list): Nombres de columnas categóricas
        
    Returns:
        tuple: (X_train_encoded, X_test_encoded, encoders_dict)
    """
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoders_dict = {}
    
    logger.info("Iniciando encoding de variables categóricas...")
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Fit SOLO con train
        le.fit(X_train_encoded[col].astype(str))
        
        # Transform train y test
        X_train_encoded[col] = le.transform(X_train_encoded[col].astype(str))
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))
        
        # Guardar encoder
        encoders_dict[col] = le
        
        logger.debug(f"  {col}: {len(le.classes_)} categorías únicas")
    
    logger.info(f"✓ {len(encoders_dict)} variables categóricas codificadas")
    
    return X_train_encoded, X_test_encoded, encoders_dict


def normalizar_features_numericas(X_train, X_test, numerical_cols):
    """
    Normaliza features numéricas usando StandardScaler.
    Fit SOLO con train, transform en train y test.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        X_test (pd.DataFrame): Features de prueba
        numerical_cols (list): Nombres de columnas numéricas
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit SOLO con train
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    # Transform test
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    logger.info(f"✓ {len(numerical_cols)} features numéricas normalizadas")
    logger.debug(f"  Media train (post-norm): {X_train_scaled[numerical_cols].mean().mean():.4f}")
    logger.debug(f"  Std train (post-norm): {X_train_scaled[numerical_cols].std().mean():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


def pipeline_model_preprocessing(df, target_col='price', escalar=True, random_state=42):
    """
    Ejecuta el pipeline completo de preprocesamiento para ML.
    
    Pasos:
    1. Separar features y target
    2. Dividir en train/test
    3. Identificar tipos de features
    4. Encodear variables categóricas
    5. Normalizar features numéricas (opcional)
    
    Args:
        df (pd.DataFrame): Dataset completo
        target_col (str): Nombre de la columna target
        escalar (bool): Si True, normaliza features numéricas
        random_state (int): Seed para reproducibilidad
        
    Returns:
        dict: Diccionario con información del preprocesamiento
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE PREPROCESAMIENTO PARA ML")
    logger.info("=" * 80)
    
    # Separar features y target
    X, y = separar_features_target(df, target_col)
    if X is None:
        return None
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=random_state
    )
    
    logger.info(f"Train: {len(X_train):,} registros ({100*0.8:.0f}%)")
    logger.info(f"Test: {len(X_test):,} registros ({100*0.2:.0f}%)")
    logger.info(f"Precio promedio (Train): ${y_train.mean():,.2f}")
    logger.info(f"Precio promedio (Test): ${y_test.mean():,.2f}")
    logger.info(f"Diferencia: {abs(y_train.mean() - y_test.mean()) / y_train.mean() * 100:.2f}%")
    
    # Identificar tipos
    categorical_cols, numerical_cols = identificar_tipos_features(X_train)
    
    # Encodear categóricas
    X_train, X_test, encoders_dict = encodear_categoricas(
        X_train, X_test, categorical_cols
    )
    
    # Normalizar numéricas (si aplica)
    scaler = None
    if escalar:
        X_train, X_test, scaler = normalizar_features_numericas(
            X_train, X_test, numerical_cols
        )
    
    logger.info("=" * 80)
    
    # Retornar diccionario con toda la información
    resultado = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'encoders': encoders_dict,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist()
    }
    
    return resultado