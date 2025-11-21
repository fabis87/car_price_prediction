"""
API FastAPI para predicción de precios de vehículos usados.
Expone endpoints para realizar predicciones y obtener información del modelo.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import json

from .models import (
    VehicleFeatures, 
    PricePrediction, 
    CategoryOptions, 
    HealthCheck
)
from .predictor import VehiclePricePredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Vehicle Price Prediction API",
    description="API para predecir precios de vehículos usados basado en sus características",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir requests desde el dashboard Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar el dominio del dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar predictor al iniciar la aplicación
predictor = None

@app.on_event("startup")
async def startup_event():
    """Inicializa el predictor al arrancar la API."""
    global predictor
    try:
        logger.info("Inicializando predictor...")
        predictor = VehiclePricePredictor(models_dir="src/models")
        logger.info("✓ Predictor inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar predictor: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """
    Endpoint raíz de bienvenida.
    """
    return {
        "message": "Vehicle Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check de la API.
    Verifica que el modelo esté cargado y funcionando.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está cargado"
        )
    
    model_info = predictor.get_model_info()
    
    return HealthCheck(
        status="healthy",
        model_loaded=True,
        model_type=model_info['model_type'],
        training_date=model_info['training_date'],
        api_version="1.0.0"
    )


@app.post("/vehicles/predict_price", response_model=PricePrediction, status_code=status.HTTP_200_OK)
async def predict_price(vehicle: VehicleFeatures):
    """
    Predice el precio de un vehículo basado en sus características.
    
    Args:
        vehicle: Características del vehículo (VehicleFeatures)
        
    Returns:
        PricePrediction con el precio predicho y metadatos
        
    Raises:
        HTTPException 503: Si el modelo no está disponible
        HTTPException 400: Si hay un error en la predicción
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible"
        )
    
    try:
        # Convertir el modelo Pydantic a dict
        vehicle_dict = vehicle.model_dump(by_alias=True)
        
        # Realizar predicción
        prediction = predictor.predict(vehicle_dict)
        
        logger.info(f"Predicción exitosa: {prediction['predicted_price_formatted']}")
        return PricePrediction(**prediction)
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.get("/vehicles/categories", response_model=CategoryOptions)
async def get_categories():
    """
    Obtiene todas las categorías disponibles para los campos del vehículo.
    Útil para poblar los selectboxes y sliders del dashboard.
    Lee desde categories.json (pre-generado) en lugar del CSV completo.
    
    Returns:
        CategoryOptions con listas de valores válidos para cada campo
    """
    try:
        # Leer el archivo JSON pre-generado
        categories_path = Path("src/models/categories.json")
        
        if not categories_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Archivo categories.json no encontrado. Ejecuta scripts/generate_categories.py primero."
            )
        
        with open(categories_path, 'r', encoding='utf-8') as f:
            categories_dict = json.load(f)
        
        categories = CategoryOptions(**categories_dict)
        
        logger.info("Categorías obtenidas exitosamente desde categories.json")
        return categories
        
    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear categories.json: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al parsear categories.json: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error al obtener categorías: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener categorías: {str(e)}"
        )


@app.get("/model/info", response_model=dict)
async def get_model_info():
    """
    Obtiene información detallada del modelo entrenado.
    Incluye métricas, features y metadata.
    
    Returns:
        Diccionario con información del modelo
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible"
        )
    
    try:
        model_info = predictor.get_model_info()
        logger.info("Información del modelo obtenida")
        return model_info
        
    except Exception as e:
        logger.error(f"Error al obtener info del modelo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener información del modelo: {str(e)}"
        )


@app.get("/vehicles/eda_stats")
async def get_eda_stats():
    """
    Obtiene datos agregados para visualizaciones EDA.
    Lee desde eda_data.json (pre-generado) en lugar del CSV completo.
    
    Returns:
        Diccionario con datos agregados para EDA:
        - general_stats: Estadísticas generales
        - depreciation: Datos de depreciación (precio vs edad)
        - manufacturers: Top marcas por precio
        - price_distribution: Distribución de precios
        - condition: Precio por condición
    """
    try:
        # Leer el archivo JSON pre-generado
        eda_data_path = Path("src/models/eda_data.json")
        
        if not eda_data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Archivo eda_data.json no encontrado. Ejecuta scripts/generate_eda_data.py primero."
            )
        
        with open(eda_data_path, 'r', encoding='utf-8') as f:
            eda_data = json.load(f)
        
        logger.info("Datos EDA obtenidos exitosamente desde eda_data.json")
        return eda_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear eda_data.json: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al parsear eda_data.json: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error al obtener datos EDA: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener datos EDA: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Manejador global de excepciones.
    """
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
