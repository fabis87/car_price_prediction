"""
Módulo de entrenamiento de modelos de Machine Learning.
Entrena y compara LightGBM y XGBoost.
"""

import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def entrenar_lightgbm(X_train, X_test, y_train, y_test, num_rounds=300):
    """
    Entrena un modelo LightGBM.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        X_test (pd.DataFrame): Features de prueba
        y_train (pd.Series): Target de entrenamiento
        y_test (pd.Series): Target de prueba
        num_rounds (int): Número de iteraciones
        
    Returns:
        tuple: (model, metrics)
    """
    logger.info("Entrenando LightGBM...")
    
    # Crear datasets de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parámetros
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 7,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'seed': 42,
        'num_threads': -1
    }
    
    # Entrenar
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50)
        ]
    )
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2_train': r2_score(y_train, y_train_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2_test': r2_score(y_test, y_test_pred),
    }
    
    logger.info(f"LightGBM Train - MAE: ${metrics['mae_train']:,.2f} | RMSE: ${metrics['rmse_train']:,.2f} | R²: {metrics['r2_train']:.4f}")
    logger.info(f"LightGBM Test  - MAE: ${metrics['mae_test']:,.2f} | RMSE: ${metrics['rmse_test']:,.2f} | R²: {metrics['r2_test']:.4f}")
    
    return model, metrics


def entrenar_xgboost(X_train, X_test, y_train, y_test, num_rounds=300):
    """
    Entrena un modelo XGBoost.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        X_test (pd.DataFrame): Features de prueba
        y_train (pd.Series): Target de entrenamiento
        y_test (pd.Series): Target de prueba
        num_rounds (int): Número de iteraciones
        
    Returns:
        tuple: (model, metrics)
    """
    logger.info("Entrenando XGBoost...")
    
    # Parámetros
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Entrenar
    model = xgb.XGBRegressor(
        n_estimators=num_rounds,
        early_stopping_rounds=50,
        **params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'mae_train': mean_absolute_error(y_train, y_train_pred),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2_train': r2_score(y_train, y_train_pred),
        'mae_test': mean_absolute_error(y_test, y_test_pred),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2_test': r2_score(y_test, y_test_pred),
    }
    
    logger.info(f"XGBoost Train - MAE: ${metrics['mae_train']:,.2f} | RMSE: ${metrics['rmse_train']:,.2f} | R²: {metrics['r2_train']:.4f}")
    logger.info(f"XGBoost Test  - MAE: ${metrics['mae_test']:,.2f} | RMSE: ${metrics['rmse_test']:,.2f} | R²: {metrics['r2_test']:.4f}")
    
    return model, metrics


def comparar_modelos(metrics_lgb, metrics_xgb):
    """
    Compara métricas de dos modelos y determina el mejor.
    
    Args:
        metrics_lgb (dict): Métricas de LightGBM
        metrics_xgb (dict): Métricas de XGBoost
        
    Returns:
        tuple: (mejor_modelo, comparacion_df)
    """
    comparacion = pd.DataFrame({
        'Métrica': ['MAE (Train)', 'RMSE (Train)', 'R² (Train)', 
                    'MAE (Test)', 'RMSE (Test)', 'R² (Test)'],
        'LightGBM': [
            f"${metrics_lgb['mae_train']:,.2f}",
            f"${metrics_lgb['rmse_train']:,.2f}",
            f"{metrics_lgb['r2_train']:.4f}",
            f"${metrics_lgb['mae_test']:,.2f}",
            f"${metrics_lgb['rmse_test']:,.2f}",
            f"{metrics_lgb['r2_test']:.4f}"
        ],
        'XGBoost': [
            f"${metrics_xgb['mae_train']:,.2f}",
            f"${metrics_xgb['rmse_train']:,.2f}",
            f"{metrics_xgb['r2_train']:.4f}",
            f"${metrics_xgb['mae_test']:,.2f}",
            f"${metrics_xgb['rmse_test']:,.2f}",
            f"{metrics_xgb['r2_test']:.4f}"
        ]
    })
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARACIÓN DE MODELOS")
    logger.info("=" * 80)
    logger.info("\n" + comparacion.to_string(index=False))
    
    mejor_modelo = "LightGBM" if metrics_lgb['mae_test'] < metrics_xgb['mae_test'] else "XGBoost"
    logger.info(f"\n✓ Mejor modelo (menor MAE en Test): {mejor_modelo}")
    logger.info("=" * 80)
    
    return mejor_modelo, comparacion


def obtener_feature_importance(model, feature_names, top_n=15, modelo_type='xgboost'):
    """
    Obtiene la importancia de features del modelo.
    
    Args:
        model: Modelo entrenado
        feature_names (list): Nombres de las features
        top_n (int): Número de top features a retornar
        modelo_type (str): 'xgboost' o 'lightgbm'
        
    Returns:
        pd.DataFrame: DataFrame con importancia de features
    """
    if modelo_type == 'xgboost':
        importance = model.feature_importances_
    elif modelo_type == 'lightgbm':
        importance = model.feature_importance()
    else:
        logger.warning(f"Tipo de modelo '{modelo_type}' no reconocido")
        return None
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    logger.info(f"\nTop {top_n} Features ({modelo_type}):")
    logger.info("\n" + feature_importance_df.to_string(index=False))
    
    return feature_importance_df


def pipeline_model_training(X_train, X_test, y_train, y_test, feature_names):
    """
    Ejecuta el pipeline completo de entrenamiento.
    
    Entrena LightGBM y XGBoost, compara, y retorna el mejor modelo.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        X_test (pd.DataFrame): Features de prueba
        y_train (pd.Series): Target de entrenamiento
        y_test (pd.Series): Target de prueba
        feature_names (list): Nombres de las features
        
    Returns:
        dict: Información del modelo entrenado
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE ENTRENAMIENTO DE MODELOS")
    logger.info("=" * 80)
    
    # Entrenar LightGBM
    logger.info("\n[1/4] Entrenando LightGBM...")
    model_lgb, metrics_lgb = entrenar_lightgbm(X_train, X_test, y_train, y_test)
    
    # Entrenar XGBoost
    logger.info("\n[2/4] Entrenando XGBoost...")
    model_xgb, metrics_xgb = entrenar_xgboost(X_train, X_test, y_train, y_test)
    
    # Comparar
    logger.info("\n[3/4] Comparando modelos...")
    mejor_modelo, comparacion_df = comparar_modelos(metrics_lgb, metrics_xgb)
    
    # Seleccionar mejor modelo
    logger.info("\n[4/4] Obteniendo feature importance...")
    if mejor_modelo == "XGBoost":
        model_final = model_xgb
        metrics_final = metrics_xgb
        tipo_final = 'xgboost'
    else:
        model_final = model_lgb
        metrics_final = metrics_lgb
        tipo_final = 'lightgbm'
    
    feature_importance = obtener_feature_importance(model_final, feature_names, modelo_type=tipo_final)
    
    logger.info("=" * 80)
    
    # Retornar información
    resultado = {
        'model_lgb': model_lgb,
        'model_xgb': model_xgb,
        'model_final': model_final,
        'model_type': tipo_final,
        'metrics_lgb': metrics_lgb,
        'metrics_xgb': metrics_xgb,
        'metrics_final': metrics_final,
        'comparacion': comparacion_df,
        'feature_importance': feature_importance,
        'mejor_modelo': mejor_modelo
    }
    
    return resultado