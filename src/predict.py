import pandas as pd
import joblib
import numpy as np

def predict_rain(features_dict):
    """
    Predice la lluvia basándose en un diccionario de características.
    Requerimientos: 
    - mes, dia_anio
    - precipitacion_lag1, lag2, lag3
    - temperatura_lag1, lag2, lag3
    - humedad_lag1, lag2, lag3
    - presion_lag1, lag2, lag3
    - precipitacion_roll_mean_7
    """
    model = joblib.load('models/rain_model.pkl')
    
    # Convertir diccionario a DataFrame
    # Asegurando el orden de columnas con el que fue entrenado (esto es crucial en sklearn)
    # Para hacerlo robusto, deberíamos haber guardado el nombre de las columnas.
    # Asumiremos el orden basado en cómo se creó X en train_model.py
    # Pero para estar seguros, cargaremos una muestra del csv procesado para ver las col
    try:
        sample_df = pd.read_csv('data/processed_rain_data.csv', nrows=1)
        feature_cols = [col for col in sample_df.columns if col not in ['fecha', 'precipitacion']]
    except:
        print("Advertencia: No se pudo cargar processed_rain_data.csv para verificar columnas.")
        feature_cols = None # Fallback peligroso

    df = pd.DataFrame([features_dict])
    
    if feature_cols:
        # Reordenar y rellenar faltantes con 0 si es necesario
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]
    
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    print("--- Predicción de Lluvia Modelo IDEAM ---")
    # Ejemplo: Un día con alta humedad y lluvia en los días previos
    example_input = {
        'mes': 5,
        'dia_anio': 140,
        'temperatura': 24, # Actual? No, train usaba lags. Revisemos train_model.
        # En train_model.py, feature 'temperatura' se usó?
        # df[features] incluia todo menos fecha y precipitacion.
        # En preprocessing.py, 'temperatura', 'humedad', 'presion' NO se shiftaron para ser lags EXCLUSIVAMENTE?
        # Revisando preprocessing.py: 
        # df[f'temperatura_lag{lag}'] = ...
        # Pero las columnas originales 'temperatura', 'humedad', 'presion' SE MANTUVIERON en el df.
        # Por lo tanto, el modelo usa la temperatura/humedad/presion del MISMO DIA para predecir lluvia del MISMO DIA?
        # Si es así, es "Nowcasting" o diagnóstico, no pronóstico futuro puro si no tenemos esos datos.
        # Asumamos que tenemos pronóstico de temperatura/humedad para mañana y queremos saber si lloverá.
        
        'temperatura': 22.5,
        'humedad': 85.0,
        'presion': 1010.0,
        
        'precipitacion_lag1': 10.5,
        'precipitacion_lag2': 5.0,
        'precipitacion_lag3': 0.0,
        
        'temperatura_lag1': 23.0,
        'temperatura_lag2': 24.0,
        'temperatura_lag3': 24.5,
        
        'humedad_lag1': 80.0,
        'humedad_lag2': 75.0,
        'humedad_lag3': 60.0,
        
        'presion_lag1': 1012.0,
        'presion_lag2': 1013.0,
        'presion_lag3': 1014.0,
        
        'precipitacion_roll_mean_7': 5.2
    }
    
    result = predict_rain(example_input)
    print(f"\nEntradas:\nTemperatura: {example_input['temperatura']}°C")
    print(f"Humedad: {example_input['humedad']}%")
    print(f"Lluvia ayer: {example_input['precipitacion_lag1']} mm")
    print(f"\n>>> Predicción de Precipitación: {result:.2f} mm")
    
    if result > 0.5:
        print("Probabilidad de lluvia: ALTA (Se esperan lluvias)")
    else:
        print("Probabilidad de lluvia: BAJA (Tiempo seco)")

