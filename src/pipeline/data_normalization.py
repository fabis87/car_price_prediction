"""
Módulo de normalización de datos para el proyecto de predicción de precios de vehículos usados.
Estandariza tipos de datos y normaliza strings.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def convertir_tipos_datos(df):
    """
    Convierte columnas a sus tipos de datos apropiados.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con tipos de datos corregidos
    """
    df_normalizado = df.copy()
    
    # Convertir YEAR a INT
    if 'year' in df_normalizado.columns:
        df_normalizado['year'] = pd.to_numeric(df_normalizado['year'], errors='coerce').astype('Int64')
        logger.debug("YEAR convertido a Int64")
    
    # Convertir PRICE a FLOAT
    if 'price' in df_normalizado.columns:
        df_normalizado['price'] = pd.to_numeric(df_normalizado['price'], errors='coerce').astype('float64')
        logger.debug("PRICE convertido a float64")
    
    # Convertir ODOMETER a FLOAT
    if 'odometer' in df_normalizado.columns:
        df_normalizado['odometer'] = pd.to_numeric(df_normalizado['odometer'], errors='coerce').astype('float64')
        logger.debug("ODOMETER convertido a float64")
    
    logger.info("Tipos de datos convertidos exitosamente")
    return df_normalizado


def normalizar_cylinders(df):
    """
    Convierte la columna 'cylinders' de string a número.
    Extrae el número de la cadena (ej: "8 cylinders" → 8)
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con cylinders normalizado
    """
    if 'cylinders' not in df.columns:
        return df
    
    df_normalizado = df.copy()
    
    # Extraer número de la cadena
    df_normalizado['cylinders'] = (
        df_normalizado['cylinders']
        .astype(str)
        .str.extract(r'(\d+)', expand=False)
    )
    
    df_normalizado['cylinders'] = pd.to_numeric(df_normalizado['cylinders'], errors='coerce').astype('Int64')
    logger.info("CYLINDERS normalizado a números")
    
    return df_normalizado


def normalizar_drive(df):
    """
    Convierte la columna 'drive' a valores numéricos.
    Mapeo: 'fwd' → 1, 'rwd' → 2, '4wd' → 4
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con drive normalizado
    """
    if 'drive' not in df.columns:
        return df
    
    df_normalizado = df.copy()
    
    drive_mapping = {'4wd': 4, 'fwd': 1, 'rwd': 2}
    df_normalizado['drive'] = df_normalizado['drive'].map(drive_mapping)
    df_normalizado['drive'] = pd.to_numeric(df_normalizado['drive'], errors='coerce').astype('Int64')
    
    logger.info("DRIVE normalizado: fwd=1, rwd=2, 4wd=4")
    
    return df_normalizado


def normalizar_strings(df):
    """
    Normaliza columnas de texto: lowercase, strip, y reemplaza 'nan' (string) con NaN (real).
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con strings normalizados
    """
    columnas_string = [
        'manufacturer', 'model', 'condition', 'fuel', 
        'transmission', 'type', 'state'
    ]
    
    df_normalizado = df.copy()
    
    for col in columnas_string:
        if col in df_normalizado.columns:
            df_normalizado[col] = df_normalizado[col].astype(str).str.lower().str.strip()
            df_normalizado[col] = df_normalizado[col].replace('nan', np.nan)
    
    logger.info(f"Strings normalizados en {len(columnas_string)} columnas")
    
    return df_normalizado


def pipeline_normalizacion(df):
    """
    Ejecuta el pipeline completo de normalización de datos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame completamente normalizado
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE NORMALIZACIÓN")
    logger.info("=" * 80)
    
    df = convertir_tipos_datos(df)
    df = normalizar_cylinders(df)
    df = normalizar_drive(df)
    df = normalizar_strings(df)
    
    logger.info(f"Estado final: {len(df):,} filas × {len(df.columns)} columnas")
    logger.info("=" * 80)
    
    return df