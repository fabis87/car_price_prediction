# API - Vehicle Price Prediction

API desarrollada con FastAPI para predecir precios de vehículos usados.

## 🚀 Instalación y Ejecución

### 1. Instalar dependencias

```bash
pip install -r requirements-api.txt
```

### 2. Ejecutar la API

Desde la raíz del proyecto:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en: `http://localhost:8000`

### 3. Documentación interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 Endpoints

### `GET /`
Endpoint raíz de bienvenida.

**Response:**
```json
{
  "message": "Vehicle Price Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### `GET /health`
Health check de la API. Verifica que el modelo esté cargado.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "XGBoost",
  "training_date": "2025-11-15T05:10:21.097118",
  "api_version": "1.0.0"
}
```

---

### `POST /vehicles/predict_price`
**Endpoint principal**: Predice el precio de un vehículo.

**Request Body:**
```json
{
  "manufacturer": "toyota",
  "condition": "good",
  "cylinders": 6,
  "fuel": "gas",
  "odometer": 45000,
  "transmission": "automatic",
  "drive": "4wd",
  "type": "pickup",
  "state": "tx",
  "age": 5
}
```

**Response:**
```json
{
  "predicted_price": 25430.50,
  "predicted_price_formatted": "$25,430.50",
  "confidence_interval_lower": 22000.00,
  "confidence_interval_upper": 29000.00,
  "model_version": "XGBoost-v1.0"
}
```

---

### `GET /vehicles/categories`
Obtiene todas las categorías disponibles para cada campo.
**Útil para poblar los selectboxes del dashboard.**

**Response:**
```json
{
  "manufacturers": ["acura", "alfa-romeo", "audi", "bmw", ...],
  "conditions": ["salvage", "fair", "good", "excellent", "like new", "new"],
  "fuels": ["diesel", "electric", "gas", "hybrid", "other"],
  "transmissions": ["automatic", "manual", "other"],
  "drives": ["fwd", "rwd", "4wd"],
  "types": ["bus", "convertible", "coupe", "hatchback", ...],
  "states": ["al", "ar", "az", "ca", ...],
  "cylinders_range": {"min": 3, "max": 12},
  "age_range": {"min": 1, "max": 45},
  "odometer_range": {"min": 0, "max": 299999}
}
```

---

### `GET /model/info`
Obtiene información detallada del modelo entrenado.

**Response:**
```json
{
  "model_type": "XGBoost",
  "training_date": "2025-11-15T05:10:21.097118",
  "metrics": {
    "mae_train": 3503.75,
    "rmse_train": 5953.67,
    "r2_train": 0.8314,
    "mae_test": 3642.28,
    "rmse_test": 6494.89,
    "r2_test": 0.8039
  },
  "total_features": 20,
  "categorical_features": ["manufacturer", "condition", "fuel", ...],
  "numerical_features": ["cylinders", "odometer", "age", ...]
}
```

## 🧪 Pruebas

Ejecutar el script de pruebas:

```bash
python src/api/test_api.py
```

Esto probará todos los endpoints y mostrará los resultados.

## 📦 Estructura

```
src/api/
├── __init__.py          # Inicialización del módulo
├── main.py              # Aplicación FastAPI principal
├── models.py            # Modelos Pydantic (validación)
├── predictor.py         # Lógica de predicción y carga del modelo
├── test_api.py          # Script de pruebas
└── README.md            # Este archivo
```

## 🔧 Configuración

### Variables de entorno (opcional)

Crear un archivo `.env` en la raíz:

```bash
API_HOST=0.0.0.0
API_PORT=8000
MODELS_DIR=src/models
DATA_DIR=data/processed
```

## 🐛 Troubleshooting

### Error: "El modelo no está cargado"

Verifica que los archivos existan en `src/models/`:
- `xgboost_model.joblib`
- `encoders.joblib`
- `scaler.joblib`
- `metadata.json`

### Error: "Categoría desconocida"

Si ingresas una categoría que no existía en el entrenamiento, la API usará un valor por defecto (0) y mostrará un warning en los logs.

### CORS Error en el Dashboard

Si el dashboard no puede conectarse a la API, verifica que CORS esté habilitado en `main.py` (ya está configurado para permitir cualquier origen).

## 📝 Notas

- La API usa **Label Encoding** para variables categóricas (igual que en el entrenamiento)
- Las features se normalizan con **StandardScaler**
- Se crean **features derivadas** automáticamente (age, log_odometer, interacciones, etc.)
- El **intervalo de confianza** se calcula como ±1.5 × MAE del modelo

## 🚀 Despliegue

Para producción, puedes desplegar en:
- **Render**: https://render.com
- **Railway**: https://railway.app
- **Heroku**: https://heroku.com
- **AWS Lambda** (con Mangum)

Ejemplo de comando para producción:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 4
```
