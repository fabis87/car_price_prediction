"""
Módulo de limpieza de datos para el proyecto de predicción de precios de vehículos usados.
Elimina columnas innecesarias, detecta y elimina duplicados, y limpia valores atípicos.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def eliminar_columnas_innecesarias(df):
    """
    Elimina columnas que no son predictivas o tienen demasiados valores faltantes.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columnas eliminadas
    """
    columnas_a_eliminar = [
        'county',           # 100% nulos
        'id',               # Solo identificador único
        'url',              # No predictiva
        'region',           # Redundante con 'state'
        'region_url',       # Redundante
        'image_url',        # No usamos imágenes
        'VIN',              # 37.7% nulos, difícil de procesar
        'lat',              # 1.5% nulos, usar 'state' es suficiente
        'long',             # 1.5% nulos
        'posting_date',     # No relevante para precio
        'description',      # Texto libre (análisis futuro)
        'size',             # 72% de valores nulos
        'paint_color',      # 45% de valores nulos
        'title_status',     # Varianza nula 94% datos de 'clean'
    ]
    
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    df_limpio = df.drop(columns=columnas_existentes)
    
    logger.info(f"Eliminadas {len(columnas_existentes)} columnas innecesarias")
    logger.info(f"Columnas restantes: {len(df_limpio.columns)}")
    
    return df_limpio


def eliminar_duplicados(df):
    """
    Elimina filas completamente duplicadas.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame sin duplicados exactos
    """
    filas_antes = len(df)
    duplicados_exactos = df.duplicated().sum()
    
    df_sin_duplicados = df.drop_duplicates()
    filas_eliminadas = filas_antes - len(df_sin_duplicados)
    
    logger.info(f"Duplicados exactos detectados: {duplicados_exactos:,}")
    logger.info(f"Filas eliminadas: {filas_eliminadas:,} ({filas_eliminadas/filas_antes*100:.2f}%)")
    
    return df_sin_duplicados


def eliminar_filas_criticas_faltantes(df):
    """
    Elimina filas donde las variables críticas tienen valores faltantes.
    Las variables críticas son aquellas sin las cuales no se puede predecir el precio.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame sin valores faltantes en variables críticas
    """
    columnas_criticas = ['price', 'year', 'manufacturer', 'model', 'odometer']
    
    filas_antes = len(df)
    df_sin_criticas = df.dropna(subset=columnas_criticas)
    filas_eliminadas = filas_antes - len(df_sin_criticas)
    
    logger.info(f"Filas con variables críticas faltantes eliminadas: {filas_eliminadas:,} ({filas_eliminadas/filas_antes*100:.2f}%)")
    
    return df_sin_criticas


def limpiar_outliers_price(df, min_price=500, max_price=500000):
    """
    Elimina valores atípicos en la columna 'price'.
    
    Args:
        df (pd.DataFrame): DataFrame original
        min_price (int): Precio mínimo válido
        max_price (int): Precio máximo válido
        
    Returns:
        pd.DataFrame: DataFrame sin outliers en price
    """
    filas_antes = len(df)
    
    precio_cero = (df['price'] == 0).sum()
    precio_bajo = (df['price'] < min_price).sum()
    precio_alto = (df['price'] > max_price).sum()
    
    df_sin_outliers = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
    filas_eliminadas = filas_antes - len(df_sin_outliers)
    
    logger.info(f"Outliers en price: cero={precio_cero:,}, bajo={precio_bajo:,}, alto={precio_alto:,}")
    logger.info(f"Filas eliminadas: {filas_eliminadas:,} ({filas_eliminadas/filas_antes*100:.2f}%)")
    
    return df_sin_outliers


def limpiar_outliers_year(df, year_minimo=1980, year_maximo=2024):
    """
    Elimina valores atípicos en la columna 'year'.
    
    Args:
        df (pd.DataFrame): DataFrame original
        year_minimo (int): Año mínimo válido
        year_maximo (int): Año máximo válido
        
    Returns:
        pd.DataFrame: DataFrame sin outliers en year
    """
    filas_antes = len(df)
    
    autos_viejos = (df['year'] < year_minimo).sum()
    autos_futuros = (df['year'] > year_maximo).sum()
    
    df_sin_outliers = df[(df['year'] >= year_minimo) & (df['year'] <= year_maximo)]
    filas_eliminadas = filas_antes - len(df_sin_outliers)
    
    logger.info(f"Outliers en year: muy viejos={autos_viejos:,}, futuros={autos_futuros:,}")
    logger.info(f"Filas eliminadas: {filas_eliminadas:,} ({filas_eliminadas/filas_antes*100:.2f}%)")
    
    return df_sin_outliers


def limpiar_outliers_odometer(df, odometer_maximo=300000):
    """
    Elimina valores atípicos en la columna 'odometer'.
    
    Args:
        df (pd.DataFrame): DataFrame original
        odometer_maximo (int): Odómetro máximo válido en km
        
    Returns:
        pd.DataFrame: DataFrame sin outliers en odometer
    """
    filas_antes = len(df)
    
    odometer_extremo = (df['odometer'] > odometer_maximo).sum()
    
    df_sin_outliers = df[df['odometer'] <= odometer_maximo]
    filas_eliminadas = filas_antes - len(df_sin_outliers)
    
    logger.info(f"Outliers en odometer (>{odometer_maximo:,}): {odometer_extremo:,}")
    logger.info(f"Filas eliminadas: {filas_eliminadas:,} ({filas_eliminadas/filas_antes*100:.2f}%)")
    
    return df_sin_outliers


def pipeline_limpieza(df):
    """
    Ejecuta el pipeline completo de limpieza de datos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame completamente limpio
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE LIMPIEZA")
    logger.info("=" * 80)
    
    logger.info(f"\nEstado inicial: {len(df):,} filas × {len(df.columns)} columnas")
    
    df = eliminar_columnas_innecesarias(df)
    df = eliminar_duplicados(df)
    df = eliminar_filas_criticas_faltantes(df)
    df = limpiar_outliers_price(df)
    df = limpiar_outliers_year(df)
    df = limpiar_outliers_odometer(df)
    
    logger.info(f"\nEstado final: {len(df):,} filas × {len(df.columns)} columnas")
    logger.info("=" * 80)
    
    return df