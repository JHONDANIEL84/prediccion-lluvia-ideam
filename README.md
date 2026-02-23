Predicción de Precipitación - IDEAM

Proyecto de Machine Learning para la predicción de precipitación utilizando Random Forest y desplegado mediante Streamlit.

Este proyecto integra modelado predictivo, contenerización con Docker y trabajo colaborativo con GitHub.

👨‍💻 Integrantes

JHÓN DANIEL CALVACHE

DIEGO FERNANDO BOLAÑOS BUSTOS

STEFANNY IZQUIERDO RAMOS

📍 2026
🎓 Universidad Autónoma de Occidente

Descripción del Proyecto

El objetivo del proyecto es estimar la precipitación (mm) a partir de variables climáticas históricas, simulando escenarios de predicción meteorológica.

El modelo fue entrenado utilizando el algoritmo Random Forest, un método de aprendizaje supervisado basado en múltiples árboles de decisión que mejora la precisión y reduce el sobreajuste.

📊 Variables de Entrada

El modelo recibe como entrada:

Lluvia día -1

Lluvia día -2

Lluvia día -3

Promedio últimos 3 días

Promedio últimos 7 días

Mes

Evento extremo anterior

Salida del modelo:

🌧️ Precipitación estimada en milímetros (mm)

🧠 Tecnologías Utilizadas

Python 3.11+

Random Forest (Machine Learning)

Streamlit (interfaz web)

Docker (contenerización)

Git & GitHub (control de versiones)

uv (gestión moderna de dependencias)

🗂️ Estructura del Proyecto

prediccion-lluvia-ideam/
│
├── main.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .dockerignore
├── .python-version
└── README.md

🐳 Contenerización con Docker

La aplicación fue empaquetada en una imagen Docker para facilitar su despliegue y distribución.

Construcción de la imagen

docker build -t prediccion-lluvia .

Ejecutar el contenedor
docker run -p 8501:8501 prediccion-lluvia

Abrir en el navegador:
http://localhost:8501

☁️ Publicación en Docker Hub

La imagen fue publicada en Docker Hub:
jhondanielcalvache/prediccion-lluvia:latest

Proceso realizado:
docker login
docker tag prediccion-lluvia jhondanielcalvache/prediccion-lluvia:latest
docker push jhondanielcalvache/prediccion-lluvia:latest

Cualquier integrante puede ejecutarla con:
docker pull jhondanielcalvache/prediccion-lluvia:latest
docker run -p 8501:8501 jhondanielcalvache/prediccion-lluvia:latest

🚀 Ejecución Local sin Docker

Crear entorno virtual y sincronizar dependencias:
uv venv
uv sync
uv run streamlit run main.py

👥 Trabajo Colaborativo

El proyecto se gestiona mediante GitHub.

🧪 Testing Automatizado

El proyecto incluye pruebas automatizadas utilizando pytest, garantizando la correcta ejecución de funciones y la estabilidad del código.

Se utiliza uv como gestor de dependencias para la instalación y ejecución de las pruebas.

📦 Instalación de pytest

uv add pytest --dev

📁 Estructura de pruebas
prediccion-lluvia-ideam/
│
├── tests/
│   └── test_basico.py

Los archivos de prueba deben:

Comenzar con test_

Contener funciones que inicien con test_

Ejemplo:

def test_suma_simple():
    assert 2 + 2 == 4

    ▶️ Ejecutar pruebas

Desde la raíz del proyecto:
uv run pytest -v

Salida esperada:
tests/test_basico.py::test_suma_simple PASSED

Objetivo del Testing

Validar la correcta ejecución de funciones

Prevenir errores al agregar nuevas funcionalidades

Mantener estabilidad en el trabajo colaborativo

Implementar buenas prácticas de desarrollo profesional

Flujo recomendado:

Crear una rama:
git checkout -b nombre-funcionalidad

Realizar cambios:
git add .
git commit -m "Descripción del cambio"
git push origin nombre-funcionalidad

Crear Pull Request hacia main.

⚠️ No trabajar directamente sobre main.

📊 Estado del Proyecto

✅ Modelo Random Forest entrenado

✅ Aplicación web funcional

✅ Contenerizada con Docker

✅ Imagen publicada en Docker Hub

✅ Repositorio actualizado en GitHub

✅ Flujo colaborativo definido

✅ Testing automatizado con pytest

🎓 Contexto Académico

Proyecto universitario orientado a:

Implementación de modelos de Machine Learning

Despliegue de aplicaciones predictivas

Contenerización profesional

Trabajo colaborativo con control de versiones

Buenas prácticas de documentación técnica

