"""
Módulo de ingeniería de features para el proyecto de predicción de precios de vehículos usados.
Crea nuevas características derivadas y de interacción.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def crear_age(df):
    """
    Crea la característica AGE (edad del vehículo en años).
    age = año_actual - year
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columna 'age'
    """
    if 'year' not in df.columns:
        logger.warning("Columna 'year' no encontrada. No se crea 'age'.")
        return df
    
    df['age'] = 2025 - df['year']
    logger.info("Feature 'age' creada: 2025 - year")
    
    return df


def crear_condition_encoded(df):
    """
    Crea CONDITION_ENCODED: mapeo ordinal de condición del vehículo.
    Mapeo: salvage=0, fair=1, good=2, excellent=3, like new=4, new=5
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columna 'condition_encoded'
    """
    if 'condition' not in df.columns:
        logger.warning("Columna 'condition' no encontrada. No se crea 'condition_encoded'.")
        return df
    
    condition_mapping = {
        'salvage': 0,
        'fair': 1,
        'good': 2,
        'excellent': 3,
        'like new': 4,
        'new': 5
    }
    
    df['condition_encoded'] = df['condition'].map(condition_mapping)
    logger.info("Feature 'condition_encoded' creada (ordinal: 0-5)")
    
    return df


def crear_log_odometer(df):
    """
    Crea LOG_ODOMETER: transformación logarítmica del odómetro.
    log_odometer = log(1 + odometer)
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columna 'log_odometer'
    """
    if 'odometer' not in df.columns:
        logger.warning("Columna 'odometer' no encontrada. No se crea 'log_odometer'.")
        return df
    
    df['log_odometer'] = np.log1p(df['odometer'])
    logger.info("Feature 'log_odometer' creada: log(1 + odometer)")
    
    return df


def crear_mileage_per_year(df):
    """
    Crea MILEAGE_PER_YEAR: promedio de km recorridos por año.
    mileage_per_year = odometer / (age + 1)
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columna 'mileage_per_year'
    """
    if 'odometer' not in df.columns or 'age' not in df.columns:
        logger.warning("Columnas 'odometer' o 'age' no encontradas. No se crea 'mileage_per_year'.")
        return df
    
    df['mileage_per_year'] = df['odometer'] / (df['age'] + 1)
    logger.info("Feature 'mileage_per_year' creada: odometer / (age + 1)")
    
    return df


def crear_cylinders_squared(df):
    """
    Crea CYLINDERS_SQUARED: relación no-lineal de cilindros.
    cylinders_squared = cylinders ^ 2
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columna 'cylinders_squared'
    """
    if 'cylinders' not in df.columns:
        logger.warning("Columna 'cylinders' no encontrada. No se crea 'cylinders_squared'.")
        return df
    
    df['cylinders_squared'] = df['cylinders'] ** 2
    logger.info("Feature 'cylinders_squared' creada: cylinders^2")
    
    return df


def crear_interacciones(df):
    """
    Crea features de interacción entre variables.
    - age_x_condition: age * condition_encoded
    - age_x_log_odometer: age * log_odometer
    - cylinders_x_condition: cylinders * condition_encoded
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columnas de interacción
    """
    # age × condition
    if 'age' in df.columns and 'condition_encoded' in df.columns:
        df['age_x_condition'] = df['age'] * df['condition_encoded']
        logger.info("Feature 'age_x_condition' creada")
    
    # age × log(odometer)
    if 'age' in df.columns and 'log_odometer' in df.columns:
        df['age_x_log_odometer'] = df['age'] * df['log_odometer']
        logger.info("Feature 'age_x_log_odometer' creada")
    
    # cylinders × condition
    if 'cylinders' in df.columns and 'condition_encoded' in df.columns:
        df['cylinders_x_condition'] = df['cylinders'] * df['condition_encoded']
        logger.info("Feature 'cylinders_x_condition' creada")
    
    return df


def crear_binarias(df):
    """
    Crea features binarias (0/1) para características categóricas importantes.
    - is_automatic: 1 si transmission == 'automatic', 0 en caso contrario
    - is_4wd: 1 si drive == '4wd', 0 en caso contrario
    - is_rwd: 1 si drive == 'rwd', 0 en caso contrario
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columnas binarias
    """
    if 'transmission' in df.columns:
        df['is_automatic'] = (df['transmission'] == 'automatic').astype(int)
        logger.info("Feature 'is_automatic' creada")
    
    if 'drive' in df.columns:
        df['is_4wd'] = (df['drive'] == '4wd').astype(int)
        df['is_rwd'] = (df['drive'] == 'rwd').astype(int)
        logger.info("Features 'is_4wd' e 'is_rwd' creadas")
    
    return df


def seleccionar_columnas_finales(df):
    """
    Selecciona y ordena las columnas finales del dataset.
    Prioridad: target, numéricas originales, categóricas, features derivadas, binarias.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columnas seleccionadas y ordenadas
    """
    columnas_finales = [
        # Target
        'price',
        
        # Numéricas originales
        'cylinders',
        'odometer',
        
        # Categóricas originales
        'manufacturer',
        'condition',
        'fuel',
        'transmission',
        'drive',
        'type',
        'state',
        
        # Features numéricas derivadas
        'age',
        'condition_encoded',
        'log_odometer',
        'mileage_per_year',
        'cylinders_squared',
        
        # Interacciones
        'age_x_condition',
        'age_x_log_odometer',
        'cylinders_x_condition',
        
        # Binarias
        'is_automatic',
        'is_4wd',
        'is_rwd',
    ]
    
    # Filtrar solo columnas que existen
    columnas_disponibles = [col for col in columnas_finales if col in df.columns]
    
    df_final = df[columnas_disponibles].copy()
    
    logger.info(f"Columnas finales seleccionadas: {len(df_final.columns)}")
    
    return df_final


def pipeline_feature_engineering(df):
    """
    Ejecuta el pipeline completo de ingeniería de features.
    
    Crea:
    1. Features derivadas: age, condition_encoded, log_odometer, mileage_per_year, cylinders_squared
    2. Features de interacción: age×condition, age×log_odometer, cylinders×condition
    3. Features binarias: is_automatic, is_4wd, is_rwd
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con todas las features creadas y seleccionadas
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE INGENIERÍA DE FEATURES")
    logger.info("=" * 80)
    
    columnas_antes = len(df.columns)
    
    df = crear_age(df)
    df = crear_condition_encoded(df)
    df = crear_log_odometer(df)
    df = crear_mileage_per_year(df)
    df = crear_cylinders_squared(df)
    df = crear_interacciones(df)
    df = crear_binarias(df)
    
    df = seleccionar_columnas_finales(df)
    
    columnas_despues = len(df.columns)
    logger.info(f"Columnas: {columnas_antes} → {columnas_despues}")
    logger.info(f"Filas: {len(df):,}")
    logger.info(f"Valores nulos: {df.isnull().sum().sum()}")
    logger.info("=" * 80)
    
    return df