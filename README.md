# Modelo de Predicción de Lluvia con Datos Climatológicos (Tipo IDEAM) 🌧️

## 1. Descripción del Proyecto
Este proyecto implementa un sistema de Machine Learning completo para predecir la cantidad de precipitación diaria en Colombia, basado en datos meteorológicos históricos como temperatura, humedad y presión atmosférica. 

El sistema incluye un generador de datos sintéticos (para simular el formato de datos del IDEAM), un pipeline de preprocesamiento, y un modelo **Random Forest Regressor** entrenado para realizar predicciones precisas.

## 2. Estructura del Proyecto
El proyecto está organizado de la siguiente manera:

```
.
├── data/                   # Almacenamiento de datos
│   ├── rain_data.csv       # Datos crudos (generados o descargados)
│   └── processed_rain_data.csv # Datos limpios y procesados
├── models/                 # Modelos entrenados
│   └── rain_model.pkl      # Modelo Random Forest guardado
├── src/                    # Código fuente
│   ├── data_loader.py      # Generación de datos sintéticos
│   ├── preprocessing.py    # Limpieza e ingeniería de características
│   ├── train_model.py      # Entrenamiento y evaluación del modelo
│   └── predict.py          # Script de inferencia (predicción en nuevos datos)
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo
```

## 3. Requisitos Previos

Asegúrese de tener Python instalado. Instale las librerías necesarias ejecutando:

```bash
pip install -r requirements.txt
```

Las librerías principales son:
- **pandas** y **numpy**: Para manipulación de datos numéricos.
- **scikit-learn**: Para el modelo Random Forest y métricas.
- **matplotlib** y **seaborn**: Para visualización (opcional).

## 4. Cómo Ejecutar el Proyecto (Paso a Paso)

El flujo de trabajo es modular. Puede ejecutar cada paso de forma independiente:

### Paso 1: Obtención de Datos
Si no posee un archivo CSV real del IDEAM, ejecute este script para generar datos simulados realistas:

```bash
python src/data_loader.py
```
> **Output**: Genera `data/rain_data.csv`.

### Paso 2: Preprocesamiento e Ingeniería de Características
Este paso limpia los datos y crea nuevas vriables predictivas, como retardos (lags) y medias móviles (rolling means) para capturar la tendencia temporal del clima.

```bash
python src/preprocessing.py
```
> **Output**: Genera `data/processed_rain_data.csv`.

*Características generadas:*
- `precipitacion_lag1, lag2, lag3`: Lluvia de los 3 días anteriores.
- `temperatura_lag...`, `humedad_lag...`: Variables climáticas pasadas.
- `precipitacion_roll_mean_7`: Promedio de lluvia de la última semana.

### Paso 3: Entrenamiento del Modelo
Entrena el modelo **Random Forest Regressor** utilizando el 80% de los datos para entrenamiento y el 20% para validación.

```bash
python src/train_model.py
```
> **Output**: Guarda el modelo en `models/rain_model.pkl` y muestra métricas de desempeño (RMSE, R2).

### Paso 4: Realizar Predicciones
Para predecir la lluvia en un día específico, use el script de inferencia. Puede modificar los valores de entrada dentro del script.

```bash
python src/predict.py
```
> **Output**: Muestra la cantidad de lluvia esperada en milímetros (mm).

## 5. Resultados del Modelo
El modelo utiliza las siguientes variables predictoras, ordenadas por importancia (basado en el entrenamiento simulado):
1. **Humedad (Actual y Pasada)**: La variable más influyente.
2. **Temperatura**: Correlacionada inversamente con la lluvia en muchos casos.
3. **Presión Atmosférica**: Indicador de tormentas.
4. **Historia de Lluvia**: Si llovió ayer, es probable que la tendencia continúe o cese dependiendo del patrón estacional.

## 6. Uso con Datos Reales (IDEAM)
Para adaptar este proyecto a datos reales:
1. Obtenga un archivo CSV del IDEAM con columnas de Fecha, Temperatura, Humedad, Presión y Precipitación.
2. Asegúrese de renombrar las columnas para que coincidan con las esperadas por `src/preprocessing.py` (`fecha`, `temperatura`, `humedad`, `presion`, `precipitacion`).
3. Reemplace el archivo `data/rain_data.csv`.
4. Ejecute el pipeline desde el **Paso 2**.

---
**Desarrollado con Python y Scikit-Learn.**
