# 🚗 Sistema de Predicción de Precios de Vehículos Usados

Sistema completo de Machine Learning para predecir precios de vehículos usados basado en sus características. El proyecto incluye un pipeline de datos robusto, una API REST con FastAPI y un dashboard interactivo con Streamlit.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso del Proyecto](#uso-del-proyecto)
- [Docker](#docker)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [API Endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [Pipeline de Datos](#pipeline-de-datos)
- [Contribución](#contribución)

## 📖 Descripción del Proyecto

Este proyecto implementa un sistema completo de predicción de precios de vehículos usados utilizando datos de Craigslist. El sistema está compuesto por tres componentes principales:

1. **Pipeline de Datos**: Procesa y limpia los datos crudos del dataset de vehículos
2. **API FastAPI**: Servicio REST que expone el modelo entrenado para realizar predicciones
3. **Dashboard Streamlit**: Interfaz interactiva para explorar datos y realizar predicciones

### Dataset

El proyecto utiliza el dataset [Craigslist Car/Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) de Kaggle, que contiene información de vehículos usados publicados en Craigslist en Estados Unidos.

## 🏗️ Arquitectura

El sistema sigue una arquitectura desacoplada de 3 componentes:

```
┌─────────────────┐
│   Dashboard     │  (Streamlit - Puerto 8501)
│   (Frontend)    │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│   API FastAPI   │  (Puerto 8000)
│   (Backend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pipeline       │  (Procesamiento de datos)
│  + Modelo ML    │
└─────────────────┘
```

**Características clave:**
- El dashboard consume todos los datos a través de la API (no lee archivos locales)
- Pipeline modular y ejecutable independientemente
- Modelo entrenado con XGBoost/LightGBM

## 📦 Requisitos

### Software Necesario

- **Python 3.11+** (recomendado 3.11)
- **Docker** y **Docker Compose** (opcional, para ejecutar con contenedores)
- **Git** (para clonar el repositorio)

### Dependencias Python

El proyecto tiene tres archivos de requisitos:

- `requirements.txt`: Dependencias completas del proyecto
- `requirements-api.txt`: Dependencias mínimas para la API
- `requirements-dashboard.txt`: Dependencias mínimas para el dashboard

## 🚀 Instalación

### Opción 1: Instalación Local (Recomendado para desarrollo)

#### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd car_price_prediction
```

#### 2. Crear entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instalar dependencias

```bash
# Instalar todas las dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

O instalar por componente:

```bash
# Solo para API
pip install -r requirements-api.txt

# Solo para Dashboard
pip install -r requirements-dashboard.txt
```

#### 4. Descargar el dataset

1. Descarga el archivo `vehicles.csv` desde [Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
2. Colócalo en la carpeta `data/` (crea la carpeta si no existe)

```bash
mkdir -p data
# Copia vehicles.csv a data/
```

#### 5. Ejecutar el pipeline de datos

```bash
# Desde la raíz del proyecto
python -m src.pipeline.data_loader
# O ejecuta el pipeline completo desde un script principal
```

### Opción 2: Instalación con Docker (Recomendado para producción)

Ver sección [Docker](#docker) más abajo.

## 💻 Uso del Proyecto

### Activar el Entorno Virtual

Antes de ejecutar cualquier comando, asegúrate de tener el entorno virtual activado:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Ejecutar la API

#### Opción A: Con uvicorn directamente

```bash
# Desde la raíz del proyecto
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Opción B: Con Python

```bash
python -m src.api.main
```

La API estará disponible en:
- **URL Base**: http://localhost:8000
- **Documentación Swagger**: http://localhost:8000/docs
- **Documentación ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Ejecutar el Dashboard

```bash
# Desde la raíz del proyecto
streamlit run src/dashboard/app.py
```

El dashboard estará disponible en: **http://localhost:8501**

**Nota**: Asegúrate de que la API esté corriendo antes de abrir el dashboard, ya que el dashboard consume datos de la API.

### Ejecutar el Pipeline de Datos

El pipeline procesa los datos crudos y genera los archivos procesados necesarios:

```bash
# Ejecutar el pipeline completo
python -m src.pipeline.data_loader
```

## 🐳 Docker

### Requisitos Previos

- Docker Desktop instalado y corriendo
- Docker Compose (incluido en Docker Desktop)

### Uso con Docker Compose

Docker Compose es la forma más sencilla de ejecutar todo el sistema:

#### 1. Construir y levantar los servicios

```bash
docker-compose up --build
```

Este comando:
- Construye las imágenes de la API y el Dashboard
- Levanta ambos contenedores
- Configura la red entre ellos
- Expone los puertos necesarios

#### 2. Acceder a los servicios

- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501

#### 3. Ver logs

```bash
# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f api
docker-compose logs -f dashboard
```

#### 4. Detener los servicios

```bash
# Detener sin eliminar contenedores
docker-compose stop

# Detener y eliminar contenedores
docker-compose down

# Detener, eliminar contenedores y volúmenes
docker-compose down -v
```

### Uso Individual de Dockerfiles

#### Construir imagen de la API

```bash
docker build -f Dockerfile.api -t vehicle-price-api .
```

#### Ejecutar contenedor de la API

```bash
docker run -d \
  --name vehicle-price-api \
  -p 8000:8000 \
  -v $(pwd)/src/api:/app/src/api \
  -v $(pwd)/src/models:/app/src/models \
  vehicle-price-api
```

#### Construir imagen del Dashboard

```bash
docker build -f Dockerfile.dashboard -t vehicle-price-dashboard .
```

#### Ejecutar contenedor del Dashboard

```bash
docker run -d \
  --name vehicle-price-dashboard \
  -p 8501:8501 \
  -v $(pwd)/src/dashboard:/app/src/dashboard \
  -e API_URL=http://api:8000 \
  vehicle-price-dashboard
```

### Volúmenes Docker

Los Dockerfiles configuran volúmenes para desarrollo:
- `./src/api` → `/app/src/api` (API)
- `./src/dashboard` → `/app/src/dashboard` (Dashboard)
- `./src/models` → `/app/src/models` (Modelos entrenados)
- `./data/processed` → `/app/data/processed` (Datos procesados)

Esto permite editar código localmente y ver cambios en tiempo real en los contenedores.

### Health Checks

La API incluye un health check que verifica:
- Que el modelo esté cargado correctamente
- Que el servicio esté respondiendo

Puedes verificar el estado con:

```bash
curl http://localhost:8000/health
```

## 📁 Estructura del Proyecto

```
car_price_prediction/
│
├── config/                 # Configuración (logging, etc.)
│   └── logging_config.py
│
├── data/                   # Datos
│   ├── raw
│       ├── vehicles.csv        # Dataset original (descargar de Kaggle)
│   |── processed/          # Datos procesados
│       ├── vehicles_clean_imputed.csv
│       └── vehicles_with_features.csv
│
├── notebook/               # Jupyter notebooks de exploración
│   ├── exploracion.ipynb
│   ├── feature_engineering.ipynb
│   └── machine_learning.ipynb
│
├── src/
│   ├── api/                # API FastAPI
│   │   ├── main.py         # Aplicación principal
│   │   ├── models.py       # Modelos Pydantic
│   │   ├── predictor.py    # Clase predictor
│   │   └── test_api.py     # Tests
│   │
│   ├── dashboard/          # Dashboard Streamlit
│   │   └── app.py          # Aplicación principal
│   │
│   ├── models/             # Modelos entrenados
│   │   ├── xgboost_model.joblib
│   │   ├── encoders.joblib
│   │   ├── scaler.joblib
│   │   └── metadata.json
│   │
│   └── pipeline/           # Pipeline de procesamiento
│       ├── data_loader.py
│       ├── data_cleaning.py
│       ├── data_imputation.py
│       ├── data_normalization.py
│       ├── data_new_features.py
│       ├── model_preprocessing.py
│       ├── model_training.py
│       └── model_saving.py
│
├── venv/                   # Entorno virtual (no versionar)
│
├── .dockerignore           # Archivos ignorados por Docker
├── docker-compose.yml      # Configuración Docker Compose
├── Dockerfile.api          # Dockerfile para API
├── Dockerfile.dashboard    # Dockerfile para Dashboard
├── requirements.txt        # Dependencias completas
├── requirements-api.txt    # Dependencias API
├── requirements-dashboard.txt  # Dependencias Dashboard
└── README.md               # Este archivo
```

## 🔌 API Endpoints

### `GET /`
Endpoint raíz con información básica de la API.

**Respuesta:**
```json
{
  "message": "Vehicle Price Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### `GET /health`
Health check de la API. Verifica que el modelo esté cargado.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "XGBoost",
  "training_date": "2025-18-15T05:10:21.097118",
  "api_version": "1.0.0"
}
```

### `POST /vehicles/predict_price`
Predice el precio de un vehículo basado en sus características.

**Request Body:**
```json
{
  "manufacturer": "toyota",
  "condition": "excellent",
  "cylinders": 4,
  "fuel": "gas",
  "odometer": 50000,
  "transmission": "automatic",
  "drive": "fwd",
  "type": "sedan",
  "state": "ca",
  "age": 5
}
```

**Respuesta:**
```json
{
  "predicted_price": 18500.50,
  "predicted_price_formatted": "$18,500.50",
  "confidence": "high"
}
```

### `GET /vehicles/categories`
Obtiene todas las categorías disponibles para poblar los selectboxes del dashboard.

**Respuesta:**
```json
{
  "manufacturers": ["toyota", "ford", "honda", ...],
  "conditions": ["salvage", "fair", "good", "excellent", "like new", "new"],
  "fuels": ["gas", "diesel", "electric", ...],
  "transmissions": ["automatic", "manual", "other"],
  "drives": ["fwd", "rwd", "4wd"],
  "types": ["sedan", "suv", "truck", ...],
  "states": ["ca", "tx", "ny", ...],
  "cylinders_range": {"min": 3, "max": 12},
  "age_range": {"min": 0, "max": 45},
  "odometer_range": {"min": 0, "max": 300000}
}
```

### `GET /model/info`
Obtiene información detallada del modelo entrenado (métricas, features, etc.).

**Respuesta:**
```json
{
  "model_type": "XGBoost",
  "training_date": "2025-18-15T05:10:21.097118",
  "metrics": {
    "mae_train": 3503.75,
    "rmse_train": 5953.67,
    "r2_train": 0.8314,
    "mae_test": 3642.28,
    "rmse_test": 6494.89,
    "r2_test": 0.8039
  },
  "categorical_features": [...],
  "numerical_features": [...],
  "total_features": 20,
  "training_samples": 199764,
  "test_samples": 49942
}
```

### Documentación Interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📊 Dashboard

El dashboard Streamlit proporciona:

1. **Herramienta de Tasación**: Formulario interactivo para ingresar características de un vehículo y obtener una predicción de precio
2. **Visualizaciones EDA**: Gráficos exploratorios de los datos
3. **Filtros Interactivos**: Sliders y selectboxes para explorar los datos en tiempo real

### Características del Dashboard

- ✅ Consume todos los datos a través de la API (no lee archivos locales)
- ✅ Al menos 2 widgets interactivos (filtros, sliders)
- ✅ Visualizaciones que se actualizan en tiempo real
- ✅ Formulario de predicción de precios
- ✅ Gráficos de EDA (distribución de precios, depreciación, etc.)

## 🔄 Pipeline de Datos

El pipeline de datos procesa el dataset crudo en varias etapas:

1. **Carga de Datos**: Lee el archivo `vehicles.csv`
2. **Limpieza**: Elimina columnas innecesarias, duplicados y outliers
3. **Normalización**: Estandariza tipos de datos y normaliza strings
4. **Imputación**: Maneja valores faltantes
5. **Feature Engineering**: Crea nuevas características derivadas
6. **Preprocesamiento**: Prepara datos para el modelo (One-Hot Encoding, escalado)
7. **Entrenamiento**: Entrena modelos LightGBM y XGBoost
8. **Guardado**: Guarda modelos y preprocesadores

Para más detalles sobre el proceso, ver [MODELO_Y_DATASET.md](MODELO_Y_DATASET.md).

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Buenas Prácticas

- Seguir las convenciones de código existentes
- Agregar comentarios y documentación
- Escribir tests cuando sea posible
- Actualizar el README si es necesario

## 📝 Licencia

Este proyecto es parte de un curso académico. Ver archivo LICENSE para más detalles.

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

---

**Nota**: Asegúrate de tener el modelo entrenado y los datos procesados antes de ejecutar la API o el dashboard. Si no los tienes, ejecuta primero el pipeline de datos.

