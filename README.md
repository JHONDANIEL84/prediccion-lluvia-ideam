# 🌧️ Predicción de Precipitación - IDEAM

Proyecto de Machine Learning para la predicción de precipitación utilizando Random Forest y despliegue con Streamlit.

---

## 👨‍🎓 Integrantes

- **JHON DANIEL CALVACHE**
- **DIEGO FERNANDO BOLAÑOS BUSTOS**
- **STEFANNY IZQUIERDO RAMOS**

📍 2026  
🎓 Universidad Autónoma de Occidente  

---

## 📌 Descripción del Proyecto

Este proyecto desarrolla un modelo de aprendizaje automático capaz de predecir la precipitación (mm) a partir de variables meteorológicas históricas.

El sistema:

- 📊 Procesa datos históricos
- 🌲 Entrena un modelo Random Forest
- 📈 Evalúa métricas de desempeño (MAE y R²)
- 🖥️ Despliega una aplicación web interactiva con Streamlit

---

## 🧠 Modelo Utilizado

- Algoritmo: Random Forest Regressor  
- División entrenamiento/prueba: 80% / 20%  
- Métricas:
  - MAE (Mean Absolute Error)
  - R² Score  

---

## 📂 Estructura del Proyecto


prediccion-lluvia-ideam/
│
├── app.py
├── src/
│ └── train_model.py
├── data/ (no incluida en el repositorio)
├── models/ (no incluida en el repositorio)
├── pyproject.toml
└── README.md


---

# ⚙️ Instalación con UV

## 1️⃣ Instalar UV (si no lo tienes)

En Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Verificar instalación:

uv --version
2️⃣ Crear entorno virtual

Desde la raíz del proyecto:

uv venv

Activar entorno:

.venv\Scripts\activate
3️⃣ Instalar dependencias

Si usas pyproject.toml:

uv sync

O si usas requirements.txt:

uv pip install -r requirements.txt
📊 Entrenar el Modelo

Colocar el dataset dentro de la carpeta data/.

Luego ejecutar:

uv run python src/train_model.py

Esto generará el modelo entrenado en la carpeta models/.

🚀 Ejecutar la Aplicación

Una vez entrenado el modelo:

uv run streamlit run app.py

La aplicación se abrirá automáticamente en el navegador.

🔬 Tecnologías Utilizadas

Python

UV (gestor moderno de entornos y dependencias)

Pandas

Scikit-Learn

Streamlit

Git & GitHub

📚 Contexto Académico

Proyecto desarrollado como parte de la formación en Inteligencia Artificial.
Universidad Autónoma de Occidente - 2026

📌 Notas Importantes

Los datos y modelos entrenados no se incluyen en el repositorio.

Para ejecutar el proyecto es necesario contar con el dataset original.

Proyecto con fines académicos.

📜 Licencia

Uso educativo.