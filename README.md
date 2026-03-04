# Predicción de Lluvia IDEAM 🌧️

1.
Información General del Proyecto

Título del Proyecto: PREDICCIÓN DE LLUVIA IDEAM.

Asignatura: Desarrollo de Proyectos de IA.

Institución: UAO Virtual.

Integrantes:

Jhon Daniel Calvache

Diego Fernando Bolaños Bustos

Stefanny Izquierdo Ramos.

Este proyecto utiliza Machine Learning (SGDClassifier) para predecir la probabilidad de lluvia basada en datos históricos de estaciones del IDEAM en Colombia.

## 🚀 Características
- **Interfaz Web:** Construida con Streamlit para visualización y predicción manual.
- **Modelo IA:** Clasificador SGD entrenado con 7 variables meteorológicas clave.
- **Seguimiento de Experimentos:** Gestión completa del ciclo de vida del modelo con MLflow.
- **Contenerización:** Listo para desplegar con Docker.
- **Documentación de Despliegue:** Guías incluidas para DigitalOcean.

## 🛠️ Instalación Local (Recomendado: uv)
Para ejecutar este proyecto localmente usando `uv`:

1. Instala las dependencias:
   ```powershell
   uv sync
   ```
2. Ejecuta la aplicación:
   ```powershell
   uv run streamlit run app.py
   ```

## 🐳 Docker
Puedes ejecutar la aplicación sin configurar Python usando la imagen de Docker Hub:

```powershell
docker run -p 8501:8501 jhondanielcalvache/prediccion-lluvia:latest
```

## 📊 MLflow
Para revisar los experimentos y métricas de entrenamiento:

1. Inicia la UI de MLflow:
   ```powershell
   uv run mlflow ui
   ```
2. Accede a `http://localhost:5000`.

## 📂 Estructura del Proyecto
- `app.py`: Aplicación principal (Streamlit).
- `src/train_model.py`: Script de entrenamiento y registro en MLflow.
- `models/`: Contiene el modelo entrenado (`rain_model.pkl`).
- `mlruns/`: Historial de experimentos de MLflow.
- `docs/`: Guías de despliegue en DigitalOcean.

## ☁️ Despliegue
Consulta la carpeta [docs/](docs/) para ver las guías paso a paso de creación de Droplet y despliegue final.

---
Desarrollado como parte del proyecto de Predicción de Lluvia IDEAM.
