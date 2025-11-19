"""
Módulo de imputación de valores faltantes para el proyecto de predicción de precios de vehículos usados.
Utiliza estrategias híbridas: KNN para variables numéricas/categóricas relacionadas y moda agrupada para categóricas.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def imputar_categoricas_simples(df):
    """
    Imputa categorías con muy pocos valores faltantes usando moda.
    - fuel: rellenar con 'gas' (moda)
    - transmission: rellenar con 'automatic' (moda)
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con categorías simples imputadas
    """
    df_imputado = df.copy()
    
    if 'fuel' in df_imputado.columns and df_imputado['fuel'].isnull().sum() > 0:
        nulos_antes = df_imputado['fuel'].isnull().sum()
        df_imputado['fuel'] = df_imputado['fuel'].fillna('gas')
        logger.info(f"FUEL imputado: {nulos_antes:,} valores → moda='gas'")
    
    if 'transmission' in df_imputado.columns and df_imputado['transmission'].isnull().sum() > 0:
        nulos_antes = df_imputado['transmission'].isnull().sum()
        df_imputado['transmission'] = df_imputado['transmission'].fillna('automatic')
        logger.info(f"TRANSMISSION imputado: {nulos_antes:,} valores → moda='automatic'")
    
    if 'type' in df_imputado.columns and df_imputado['type'].isnull().sum() > 0:
        nulos_antes = df_imputado['type'].isnull().sum()
        df_imputado['type'] = df_imputado['type'].fillna('other')
        logger.info(f"TYPE imputado: {nulos_antes:,} valores → moda='other'")
    
    return df_imputado


def imputar_knn_cylinders_drive(df):
    """
    Imputa CYLINDERS y DRIVE usando KNN basado en variables similares.
    Utiliza: year, odometer, manufacturer, model como referencia.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con cylinders y drive imputados vía KNN
    """
    df_imputado = df.copy()
    
    cols_for_knn = ['year', 'odometer', 'manufacturer', 'model', 'cylinders', 'drive']
    
    # Verificar que todas las columnas existen
    cols_disponibles = [col for col in cols_for_knn if col in df_imputado.columns]
    if len(cols_disponibles) < len(cols_for_knn):
        logger.warning(f"No todas las columnas para KNN están disponibles. Usando: {cols_disponibles}")
        return df_imputado
    
    df_knn = df_imputado[cols_for_knn].copy()
    
    nulos_antes_cyl = df_knn['cylinders'].isnull().sum()
    nulos_antes_drv = df_knn['drive'].isnull().sum()
    
    # Codificar variables categóricas
    le_manu = LabelEncoder()
    le_model = LabelEncoder()
    
    df_knn['manufacturer'] = le_manu.fit_transform(
        df_knn['manufacturer'].fillna('unknown')
    )
    df_knn['model'] = le_model.fit_transform(
        df_knn['model'].fillna('unknown')
    )
    
    # Aplicar KNN
    knn = KNNImputer(n_neighbors=5, weights='distance')
    df_knn[cols_for_knn] = knn.fit_transform(df_knn[cols_for_knn])
    
    # Redondear y convertir a enteros
    df_knn['cylinders'] = df_knn['cylinders'].round().astype(int)
    df_knn['drive'] = df_knn['drive'].round().astype(int)
    
    # Mapear drive de vuelta a categorías: 1→fwd, 2→rwd, 4→4wd
    drive_reverse_mapping = {1: 'fwd', 2: 'rwd', 4: '4wd'}
    df_knn['drive'] = df_knn['drive'].map(drive_reverse_mapping)
    
    # Asignar de vuelta al dataframe principal
    df_imputado['cylinders'] = df_knn['cylinders']
    df_imputado['drive'] = df_knn['drive']
    
    nulos_despues_cyl = df_imputado['cylinders'].isnull().sum()
    nulos_despues_drv = df_imputado['drive'].isnull().sum()
    
    logger.info(f"KNN CYLINDERS: {nulos_antes_cyl:,} → {nulos_despues_cyl:,} nulos")
    logger.info(f"KNN DRIVE: {nulos_antes_drv:,} → {nulos_despues_drv:,} nulos")
    
    return df_imputado


def imputar_condition_agrupada(df):
    """
    Imputa CONDITION usando moda agrupada por manufacturer y fuel.
    Si aún quedan nulos, usa la moda global.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con condition imputado
    """
    df_imputado = df.copy()
    
    if 'condition' not in df_imputado.columns:
        return df_imputado
    
    nulos_antes = df_imputado['condition'].isnull().sum()
    
    if nulos_antes == 0:
        return df_imputado
    
    # Agrupar por manufacturer y fuel, rellenar con moda
    df_imputado['condition'] = df_imputado.groupby(['manufacturer', 'fuel'])['condition'].transform(
        lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else 'good')
    )
    
    # Rellenar nulos restantes con 'good' (moda global)
    df_imputado['condition'] = df_imputado['condition'].fillna('good')
    
    nulos_despues = df_imputado['condition'].isnull().sum()
    logger.info(f"CONDITION imputado (agrupada): {nulos_antes:,} → {nulos_despues:,} nulos")
    
    return df_imputado


def imputar_drive_final(df):
    """
    Imputa valores restantes en DRIVE usando la moda global.
    Se ejecuta después de KNN para completar cualquier valor faltante.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con drive completamente imputado
    """
    df_imputado = df.copy()
    
    if 'drive' not in df_imputado.columns:
        return df_imputado
    
    nulos_antes = df_imputado['drive'].isnull().sum()
    
    if nulos_antes > 0:
        moda_drive = df_imputado['drive'].mode()[0] if len(df_imputado['drive'].mode()) > 0 else '4wd'
        df_imputado['drive'] = df_imputado['drive'].fillna(moda_drive)
        logger.info(f"DRIVE final imputado: {nulos_antes:,} valores → moda='{moda_drive}'")
    
    return df_imputado


def verificar_nulos_finales(df):
    """
    Verifica que no haya valores nulos restantes.
    
    Args:
        df (pd.DataFrame): DataFrame
        
    Returns:
        bool: True si no hay nulos, False en caso contrario
    """
    nulos_totales = df.isnull().sum().sum()
    
    if nulos_totales == 0:
        logger.info("✓ Verificación: NO HAY VALORES NULOS")
        return True
    else:
        nulos_por_col = df.isnull().sum()
        nulos_con_valores = nulos_por_col[nulos_por_col > 0]
        logger.warning(f"Columnas con valores nulos aún presentes:\n{nulos_con_valores}")
        return False


def pipeline_imputacion(df):
    """
    Ejecuta el pipeline completo de imputación de valores faltantes.
    Estrategia híbrida:
    1. Imputación simple (moda) para categorías con pocos nulos
    2. KNN para cylinders y drive
    3. Moda agrupada para condition
    4. Moda global para drive (nulos restantes)
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame completamente imputado
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE IMPUTACIÓN")
    logger.info("=" * 80)
    
    nulos_iniciales = df.isnull().sum().sum()
    logger.info(f"Nulos iniciales: {nulos_iniciales:,}")
    
    df = imputar_categoricas_simples(df)
    df = imputar_knn_cylinders_drive(df)
    df = imputar_condition_agrupada(df)
    df = imputar_drive_final(df)
    
    nulos_finales = df.isnull().sum().sum()
    logger.info(f"Nulos finales: {nulos_finales:,}")
    
    verificar_nulos_finales(df)
    logger.info("=" * 80)
    
    return df