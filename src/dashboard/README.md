# 📊 Dashboard Streamlit - Tasación de Vehículos

Dashboard interactivo para predicción de precios de vehículos usados. Consume la API FastAPI.

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements-dashboard.txt
```

### 2. Asegurarse de que la API esté corriendo

```bash
# En una terminal
uvicorn src.api.main:app --reload --port 8000
```

### 3. Iniciar el dashboard

```bash
streamlit run src/dashboard/app.py
```

O usar el script:
```bash
./start_dashboard.sh
```

El dashboard estará disponible en: **http://localhost:8501**

---

## 🎯 Características

### ✅ Página de Predicción
- **Formulario interactivo** con selectboxes y sliders
- **3 columnas** de inputs organizados por categoría
- **Validación automática** de datos vía API
- **Visualización del resultado** con intervalo de confianza
- **Gráfico de barras** del precio predicho
- **Resumen expandible** del vehículo ingresado

### ✅ Página de Análisis Exploratorio (EDA)
- **Curva de depreciación**: Precio vs Edad
- **Precio por marca**: Top N marcas más caras
- **Distribución de precios**: Histograma
- **Precio por condición**: Box plots
- **Métricas generales**: Total vehículos, precios promedio, etc.

### ✅ Sidebar
- **Información del modelo**: R², MAE, RMSE
- **Navegación**: Switch entre Predicción y EDA
- **Features usadas**: Resumen de variables

---

## 📋 Estructura

```
src/dashboard/
├── __init__.py       # Inicialización
├── app.py            # Aplicación principal
└── README.md         # Este archivo
```

---

## 🎨 Componentes del Dashboard

### Formulario de Predicción

**Columna 1 - Información Básica:**
- Marca del vehículo (selectbox)
- Condición (selectbox)
- Tipo de vehículo (selectbox)
- Estado (selectbox)

**Columna 2 - Motor y Transmisión:**
- Cilindros (slider)
- Combustible (selectbox)
- Transmisión (selectbox)
- Tracción (selectbox)

**Columna 3 - Uso y Antigüedad:**
- Edad en años (slider)
- Kilometraje en millas (number input)

### Visualizaciones EDA

1. **Curva de Depreciación**
   - Muestra cómo el precio disminuye con la edad
   - Línea de precio promedio y mediano

2. **Precio por Marca**
   - Top N marcas más caras
   - Gráfico de barras horizontal

3. **Distribución de Precios**
   - Histograma de frecuencias
   - Percentiles

4. **Precio por Condición**
   - Box plots por condición del vehículo
   - Comparación de rangos

---

## 🔧 Configuración

### Variables de Entorno

Crear archivo `.streamlit/secrets.toml`:

```toml
API_URL = "http://localhost:8000"
```

O editar directamente en `app.py`:
```python
API_URL = "http://localhost:8000"
```

### Para Despliegue en Streamlit Cloud

1. Subir código a GitHub
2. Conectar repositorio en Streamlit Cloud
3. Configurar secrets:
   - `API_URL = "https://tu-api-desplegada.com"`
4. Deploy automático

---

## 📊 Uso

### Hacer una Predicción

1. Ir a la página **"🔍 Predicción"**
2. Llenar el formulario con las características del vehículo
3. Hacer clic en **"🔍 PREDECIR PRECIO"**
4. Ver el resultado con el intervalo de confianza

### Ver Análisis Exploratorio

1. Ir a la página **"📈 Análisis Exploratorio"**
2. Explorar los 4 tabs de visualizaciones
3. Ajustar el slider de "Top N" en el gráfico de marcas

---

## 🎯 Cumplimiento de Requerimientos

✅ **Arquitectura desacoplada**: Dashboard consume API, no lee archivos locales para predicción
✅ **Interactividad**: Selectboxes, sliders, tabs, botones
✅ **Storytelling**: EDA muestra insights clave (depreciación, marcas, condición)
✅ **Widgets interactivos**: +10 widgets (selectboxes, sliders, tabs, etc.)
✅ **Actualización en tiempo real**: Predicciones on-demand vía API

---

## 🐛 Troubleshooting

### Error: "No se pudo conectar con la API"

**Solución:** Verificar que la API esté corriendo
```bash
curl http://localhost:8000/health
```

### Error: "No se pudieron cargar los datos para EDA"

**Causa:** Falta el archivo de datos procesados

**Solución:** Asegurarse de que existe:
```bash
data/processed/vehicles_with_features.csv
```

### El dashboard se ve diferente

**Causa:** Versión antigua de Streamlit

**Solución:** Actualizar
```bash
pip install --upgrade streamlit
```

---

## 📦 Dependencias

Ver `requirements-dashboard.txt`:
- streamlit
- requests
- pandas
- plotly

---

## 🚀 Despliegue en Producción

### Streamlit Community Cloud

1. Push código a GitHub
2. Ir a https://share.streamlit.io
3. Conectar repositorio
4. Seleccionar `src/dashboard/app.py`
5. Configurar secrets (API_URL)
6. Deploy

### Render.com

Crear `render.yaml`:
```yaml
services:
  - type: web
    name: vehicle-dashboard
    env: python
    buildCommand: pip install -r requirements-dashboard.txt
    startCommand: streamlit run src/dashboard/app.py --server.port $PORT
```

---

## 💡 Mejoras Futuras

- [ ] Comparación de múltiples vehículos
- [ ] Historial de predicciones
- [ ] Exportar resultados a PDF
- [ ] Filtros interactivos en EDA
- [ ] Mapa de precios por estado

---

**Dashboard listo para usar! 🎉**
