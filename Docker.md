
## 馃惓 Docker

### Requisitos Previos

- Docker Desktop instalado y corriendo
- Docker Compose (incluido en Docker Desktop)

### Uso con Docker Compose

Docker Compose es la forma m谩s sencilla de ejecutar todo el sistema:

#### 1. Construir y levantar los servicios

```bash
docker-compose up --build
```

Este comando:
- Construye las im谩genes de la API y el Dashboard
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

# Ver logs de un servicio espec铆fico
docker-compose logs -f api
docker-compose logs -f dashboard
```

#### 4. Detener los servicios

```bash
# Detener sin eliminar contenedores
docker-compose stop

# Detener y eliminar contenedores
docker-compose down

# Detener, eliminar contenedores y vol煤menes
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

### Vol煤menes Docker

Los Dockerfiles configuran vol煤menes para desarrollo:
- `./src/api` 鈫?`/app/src/api` (API)
- `./src/dashboard` 鈫?`/app/src/dashboard` (Dashboard)
- `./src/models` 鈫?`/app/src/models` (Modelos entrenados)
- `./data/processed` 鈫?`/app/data/processed` (Datos procesados)

Esto permite editar c贸digo localmente y ver cambios en tiempo real en los contenedores.

### Health Checks

La API incluye un health check que verifica:
- Que el modelo est茅 cargado correctamente
- Que el servicio est茅 respondiendo

Puedes verificar el estado con:

```bash
curl http://localhost:8000/health
```
