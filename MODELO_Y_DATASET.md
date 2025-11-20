# 📊 Documentación del Modelo y Dataset

Este documento describe en detalle el modelo de Machine Learning, el dataset utilizado, las métricas de rendimiento, las columnas, y los procesos de limpieza y normalización aplicados.

## 📦 Dataset

### Fuente

El dataset utilizado es **Craigslist Car/Trucks Data** de Kaggle:
- **Link**: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
- **Archivo**: `vehicles.csv`
- **Tamaño original**: ~426,880 registros
- **Columnas originales**: 26

### Contexto

Craigslist es la colección más grande del mundo de vehículos usados en venta. Este dataset contiene información de todas las entradas de vehículos usados dentro de Estados Unidos publicadas en Craigslist. Los datos se actualizan cada pocos meses e incluyen información relevante como precio, condición, fabricante, ubicación (latitud/longitud), y otras 18 categorías.

### Columnas Originales del Dataset

El dataset original (`vehicles.csv`) contiene las siguientes 26 columnas:

| # | Columna | Tipo | Descripción | % Nulos |
|---|---------|------|-------------|---------|
| 1 | `id` | int64 | Identificador único | 0% |
| 2 | `url` | object | URL del anuncio | 0% |
| 3 | `region` | object | Región geográfica | 0% |
| 4 | `region_url` | object | URL de la región | 0% |
| 5 | `price` | int64 | Precio del vehículo (target) | 0% |
| 6 | `year` | float64 | Año del vehículo | 0.3% |
| 7 | `manufacturer` | object | Fabricante/Marca | 4.1% |
| 8 | `model` | object | Modelo del vehículo | 1.2% |
| 9 | `condition` | object | Condición (excellent, good, fair, etc.) | 40.8% |
| 10 | `cylinders` | object | Número de cilindros | 41.6% |
| 11 | `fuel` | object | Tipo de combustible | 0.7% |
| 12 | `odometer` | float64 | Kilometraje en millas | 1.0% |
| 13 | `title_status` | object | Estado del título | 1.9% |
| 14 | `transmission` | object | Tipo de transmisión | 0.6% |
| 15 | `VIN` | object | Número de identificación del vehículo | 37.7% |
| 16 | `drive` | object | Tipo de tracción (fwd, rwd, 4wd) | 30.5% |
| 17 | `size` | object | Tamaño del vehículo | 71.8% |
| 18 | `type` | object | Tipo de vehículo (sedan, suv, truck, etc.) | 21.7% |
| 19 | `paint_color` | object | Color de la pintura | 30.5% |
| 20 | `image_url` | object | URL de la imagen | 0.02% |
| 21 | `description` | object | Descripción del vehículo | 0.02% |
| 22 | `county` | float64 | Condado | 100% |
| 23 | `state` | object | Estado (código de 2 letras) | 0% |
| 24 | `lat` | float64 | Latitud | 1.5% |
| 25 | `long` | float64 | Longitud | 1.5% |
| 26 | `posting_date` | object | Fecha de publicación | 0.02% |

## 🧹 Proceso de Limpieza de Datos

### 1. Eliminación de Columnas Innecesarias

Se eliminaron las siguientes columnas por las razones indicadas:

| Columna | Razón de Eliminación |
|---------|---------------------|
| `county` | 100% de valores nulos |
| `id` | Solo identificador único, no predictivo |
| `url` | No predictiva |
| `region` | Redundante con `state` |
| `region_url` | Redundante |
| `image_url` | No se utilizan imágenes en el modelo |
| `VIN` | 37.7% nulos, difícil de procesar |
| `lat`, `long` | 1.5% nulos, usar `state` es suficiente |
| `posting_date` | No relevante para precio |
| `description` | Texto libre (análisis futuro con NLP) |
| `size` | 72% de valores nulos |
| `paint_color` | 45% de valores nulos |
| `title_status` | Varianza nula (94% datos de 'clean') |

**Resultado**: Se eliminaron 13 columnas, quedando 13 columnas relevantes.

### 2. Eliminación de Duplicados

- Se eliminaron filas completamente duplicadas
- Se detectaron y eliminaron duplicados exactos

### 3. Eliminación de Filas con Valores Críticos Faltantes

Se eliminaron filas donde las siguientes variables críticas tenían valores faltantes:
- `price` (target)
- `year`
- `manufacturer`
- `model`
- `odometer`

Estas variables son esenciales para la predicción y no pueden ser imputadas de manera confiable.

### 4. Limpieza de Outliers

#### Precio (`price`)
- **Eliminados**: Precios = 0, precios < $500, precios > $500,000
- **Rango válido**: $500 - $500,000

#### Año (`year`)
- **Eliminados**: Años < 1980, años > 2024
- **Rango válido**: 1980 - 2024

#### Odómetro (`odometer`)
- **Eliminados**: Valores > 300,000 millas
- **Rango válido**: 0 - 300,000 millas

### Resultados del Proceso de Limpieza

- **Filas iniciales**: ~426,880
- **Filas después de limpieza**: ~249,706
- **Reducción**: ~41.5% de los datos
- **Columnas finales**: 13 (después de eliminar columnas innecesarias)

## 🔄 Proceso de Normalización

### 1. Conversión de Tipos de Datos

- **`year`**: Convertido a `Int64` (enteros con nulos)
- **`price`**: Convertido a `float64`
- **`odometer`**: Convertido a `float64`

### 2. Normalización de `cylinders`

La columna `cylinders` venía como string (ej: "8 cylinders", "4 cylinders").
- Se extrajo el número de la cadena usando expresiones regulares
- Se convirtió a `Int64`

**Ejemplo**: `"8 cylinders"` → `8`

### 3. Normalización de `drive`

Se mapeó a valores numéricos:
- `'fwd'` → `1`
- `'rwd'` → `2`
- `'4wd'` → `4`

### 4. Normalización de Strings

Se normalizaron las siguientes columnas categóricas:
- `manufacturer`
- `model`
- `condition`
- `fuel`
- `transmission`
- `type`
- `state`

**Proceso**:
1. Conversión a minúsculas
2. Eliminación de espacios en blanco al inicio y final
3. Reemplazo de strings 'nan' con valores NaN reales

## 🔧 Ingeniería de Features

### Features Derivadas

Se crearon las siguientes características nuevas:

#### 1. `age` (Edad del vehículo)
- **Cálculo**: `age = 2025 - year`
- **Tipo**: Numérica entera
- **Rango**: 0 - 45 años

#### 2. `condition_encoded` (Condición codificada ordinalmente)
- **Mapeo**:
  - `salvage` → 0
  - `fair` → 1
  - `good` → 2
  - `excellent` → 3
  - `like new` → 4
  - `new` → 5

#### 3. `log_odometer` (Transformación logarítmica del odómetro)
- **Cálculo**: `log_odometer = log(1 + odometer)`
- **Propósito**: Reducir el impacto de valores extremos

#### 4. `mileage_per_year` (Kilometraje promedio por año)
- **Cálculo**: `mileage_per_year = odometer / (age + 1)`
- **Propósito**: Capturar el uso intensivo del vehículo

#### 5. `cylinders_squared` (Relación no-lineal de cilindros)
- **Cálculo**: `cylinders_squared = cylinders²`
- **Propósito**: Capturar relaciones no lineales

### Features de Interacción

Se crearon interacciones entre variables:

#### 1. `age_x_condition`
- **Cálculo**: `age × condition_encoded`
- **Propósito**: Capturar cómo la edad afecta el precio según la condición

#### 2. `age_x_log_odometer`
- **Cálculo**: `age × log_odometer`
- **Propósito**: Capturar la relación entre edad y uso

#### 3. `cylinders_x_condition`
- **Cálculo**: `cylinders × condition_encoded`
- **Propósito**: Capturar cómo los cilindros afectan el precio según la condición

### Features Binarias

Se crearon variables binarias (0/1):

#### 1. `is_automatic`
- **Valor**: `1` si `transmission == 'automatic'`, `0` en caso contrario

#### 2. `is_4wd`
- **Valor**: `1` si `drive == '4wd'`, `0` en caso contrario

#### 3. `is_rwd`
- **Valor**: `1` si `drive == 'rwd'`, `0` en caso contrario

### Columnas Finales del Dataset Procesado

Después de la ingeniería de features, el dataset final contiene **21 columnas**:

#### Target
- `price`: Precio del vehículo (variable objetivo)

#### Numéricas Originales
- `cylinders`: Número de cilindros
- `odometer`: Kilometraje en millas

#### Categóricas Originales
- `manufacturer`: Fabricante/Marca
- `condition`: Condición del vehículo
- `fuel`: Tipo de combustible
- `transmission`: Tipo de transmisión
- `drive`: Tipo de tracción
- `type`: Tipo de vehículo
- `state`: Estado (código de 2 letras)

#### Features Numéricas Derivadas
- `age`: Edad del vehículo
- `condition_encoded`: Condición codificada (0-5)
- `log_odometer`: Logaritmo del odómetro
- `mileage_per_year`: Kilometraje promedio por año
- `cylinders_squared`: Cilindros al cuadrado

#### Features de Interacción
- `age_x_condition`: Edad × Condición
- `age_x_log_odometer`: Edad × Log(Odómetro)
- `cylinders_x_condition`: Cilindros × Condición

#### Features Binarias
- `is_automatic`: Es automático (0/1)
- `is_4wd`: Es tracción 4WD (0/1)
- `is_rwd`: Es tracción trasera (0/1)

## 🤖 Modelo de Machine Learning

### Algoritmos Evaluados

Se entrenaron y compararon dos algoritmos:

1. **LightGBM** (Light Gradient Boosting Machine)
2. **XGBoost** (Extreme Gradient Boosting)

### Modelo Seleccionado

**XGBoost** fue seleccionado como modelo final por tener el menor error absoluto medio (MAE) en el conjunto de prueba.

### Parámetros del Modelo XGBoost

```python
{
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 300,
    'early_stopping_rounds': 50
}
```

### Preprocesamiento para el Modelo

#### 1. One-Hot Encoding
Se aplicó One-Hot Encoding a las siguientes variables categóricas:
- `manufacturer`
- `condition`
- `fuel`
- `transmission`
- `drive`
- `type`
- `state`

#### 2. Escalado
Se aplicó escalado estándar (StandardScaler) a las variables numéricas.

#### 3. División de Datos
- **Train**: 80% de los datos (~199,764 muestras)
- **Test**: 20% de los datos (~49,942 muestras)
- **Random State**: 42 (para reproducibilidad)

### Features Utilizadas en el Modelo

El modelo utiliza **20 features** después del preprocesamiento:

**Categóricas (One-Hot Encoded)**:
- `manufacturer` (42 categorías)
- `condition` (6 categorías)
- `fuel` (5 categorías)
- `transmission` (3 categorías)
- `drive` (3 categorías)
- `type` (13 categorías)
- `state` (51 categorías)

**Numéricas**:
- `cylinders`
- `odometer`
- `age`
- `condition_encoded`
- `log_odometer`
- `mileage_per_year`
- `cylinders_squared`
- `age_x_condition`
- `age_x_log_odometer`
- `cylinders_x_condition`
- `is_automatic`
- `is_4wd`
- `is_rwd`

## 📈 Métricas de Rendimiento

### Métricas del Modelo Final (XGBoost)

#### Conjunto de Entrenamiento (Train)

| Métrica | Valor |
|---------|-------|
| **MAE** (Mean Absolute Error) | $3,503.75 |
| **RMSE** (Root Mean Squared Error) | $5,953.67 |
| **R²** (Coefficient of Determination) | 0.8314 |

#### Conjunto de Prueba (Test)

| Métrica | Valor |
|---------|-------|
| **MAE** (Mean Absolute Error) | $3,642.28 |
| **RMSE** (Root Mean Squared Error) | $6,494.89 |
| **R²** (Coefficient of Determination) | 0.8039 |

### Interpretación de las Métricas

#### MAE (Error Absoluto Medio)
- **Train**: $3,503.75 → El modelo predice con un error promedio de ~$3,500 en entrenamiento
- **Test**: $3,642.28 → El modelo predice con un error promedio de ~$3,600 en prueba
- **Diferencia**: ~$140, lo que indica un buen ajuste sin sobreajuste significativo

#### RMSE (Raíz del Error Cuadrático Medio)
- **Train**: $5,953.67
- **Test**: $6,494.89
- **Interpretación**: Penaliza más los errores grandes. Un RMSE de ~$6,500 indica que algunos errores pueden ser mayores, pero en promedio el modelo funciona bien.

#### R² (Coeficiente de Determinación)
- **Train**: 0.8314 → El modelo explica el 83.14% de la varianza en entrenamiento
- **Test**: 0.8039 → El modelo explica el 80.39% de la varianza en prueba
- **Interpretación**: Un R² de 0.80 indica un buen ajuste del modelo. El modelo captura la mayoría de la variabilidad en los precios.

### Comparación con LightGBM

| Métrica | LightGBM (Test) | XGBoost (Test) | Mejor |
|---------|----------------|----------------|-------|
| **MAE** | ~$3,700 | $3,642.28 | XGBoost |
| **RMSE** | ~$6,600 | $6,494.89 | XGBoost |
| **R²** | ~0.80 | 0.8039 | XGBoost |

**Conclusión**: XGBoost obtuvo mejores resultados en todas las métricas, por lo que fue seleccionado como modelo final.

## 📊 Importancia de Features

Las features más importantes según el modelo XGBoost (top 10):

1. **`age`**: La edad del vehículo es el factor más importante
2. **`odometer`**: El kilometraje tiene un impacto significativo
3. **`condition_encoded`**: La condición del vehículo es crucial
4. **`log_odometer`**: La transformación logarítmica del odómetro
5. **`manufacturer`**: La marca del vehículo influye en el precio
6. **`cylinders`**: El número de cilindros afecta el precio
7. **`mileage_per_year`**: El uso promedio anual
8. **`age_x_log_odometer`**: Interacción entre edad y uso
9. **`type`**: El tipo de vehículo (sedan, suv, truck, etc.)
10. **`state`**: La ubicación geográfica

## 💾 Archivos del Modelo

El modelo entrenado se guarda en `src/models/` con los siguientes archivos:

- **`xgboost_model.joblib`**: Modelo XGBoost entrenado
- **`encoders.joblib`**: Encoders (One-Hot Encoding) para variables categóricas
- **`scaler.joblib`**: Escalador (StandardScaler) para variables numéricas
- **`metadata.json`**: Metadatos del modelo (métricas, fecha de entrenamiento, features, etc.)

## 🔄 Flujo Completo del Pipeline

```
vehicles.csv (Datos crudos)
    ↓
[1. Carga de Datos]
    ↓
[2. Limpieza]
    - Eliminar columnas innecesarias
    - Eliminar duplicados
    - Eliminar filas con valores críticos faltantes
    - Limpiar outliers (price, year, odometer)
    ↓
[3. Normalización]
    - Convertir tipos de datos
    - Normalizar cylinders, drive
    - Normalizar strings
    ↓
[4. Imputación]
    - Manejar valores faltantes restantes
    ↓
[5. Feature Engineering]
    - Crear features derivadas (age, condition_encoded, etc.)
    - Crear features de interacción
    - Crear features binarias
    ↓
vehicles_with_features.csv (Datos procesados)
    ↓
[6. Preprocesamiento]
    - One-Hot Encoding de categóricas
    - Escalado de numéricas
    - División train/test
    ↓
[7. Entrenamiento]
    - Entrenar LightGBM
    - Entrenar XGBoost
    - Comparar modelos
    - Seleccionar mejor modelo
    ↓
[8. Guardado]
    - Guardar modelo
    - Guardar encoders
    - Guardar scaler
    - Guardar metadata
    ↓
Modelo listo para producción
```

## 📝 Notas Adicionales

### Limitaciones del Modelo

1. **Rango de Precios**: El modelo está entrenado con precios entre $500 y $500,000. Predicciones fuera de este rango pueden ser menos confiables.

2. **Años**: El modelo está entrenado con vehículos de 1980 a 2024. Vehículos más antiguos o futuros pueden tener predicciones menos precisas.

3. **Ubicación**: El modelo utiliza el estado (state) pero no la latitud/longitud específica. Esto puede limitar la precisión en áreas con variaciones de precio significativas dentro del mismo estado.

4. **Valores Faltantes**: El modelo requiere que todas las características estén presentes. Si faltan valores, se aplica imputación, pero esto puede afectar la precisión.

### Mejoras Futuras

1. **NLP en Descripciones**: Analizar las descripciones de texto para extraer características adicionales
2. **Features Geográficas**: Utilizar latitud/longitud para crear features de ubicación más granulares
3. **Modelos Ensemble**: Combinar múltiples modelos para mejorar la precisión
4. **Validación Cruzada**: Implementar k-fold cross-validation para una evaluación más robusta
5. **Hiperparámetros**: Optimización de hiperparámetros con GridSearch o Bayesian Optimization

---

**Última actualización**: Enero 2025  
**Versión del modelo**: 1.0.0

