import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(start_date='2020-01-01', end_date='2023-12-31', output_file='data/rain_data.csv'):
    """
    Genera datos meteorológicos sintéticos simulando el clima de una región tropical (Colombia).
    """
    print(f"Generando datos desde {start_date} hasta {end_date}...")
    
    # Rango de fechas
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(date_range)
    
    # Semilla para reproducibilidad
    np.random.seed(42)
    
    # Simulación de variables
    # Temperatura: Media 25°C, variabilidad normal
    temperature = np.random.normal(loc=25, scale=3, size=n)
    
    # Humedad: Media 70%, variabilidad alta
    humidity = np.random.normal(loc=70, scale=10, size=n)
    humidity = np.clip(humidity, 30, 100) # Limitar entre 30% y 100%
    
    # Presión atmosférica: Media 1013 hPa
    pressure = np.random.normal(loc=1013, scale=5, size=n)
    
    # Precipitación (Target): Dependiente de humedad y temperatura + aleatoriedad
    # Mas probabilidad de lluvia si humedad alta y temperatura baja/media
    rain_prob = (humidity - 50) / 50.0  # Probabilidad base
    rain_prob = np.clip(rain_prob, 0, 1)
    
    # Generar lluvia (mm)
    precipitation = np.zeros(n)
    for i in range(n):
        if np.random.random() < rain_prob[i]:
            # Si llueve, cantidad aleatoria (distribución gamma es común para lluvia)
            precipitation[i] = np.random.gamma(shape=2, scale=5)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'fecha': date_range,
        'temperatura': temperature,
        'humedad': humidity,
        'presion': pressure,
        'precipitacion': precipitation
    })
    
    # Asegurar directorio
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Guardar CSV
    df.to_csv(output_file, index=False)
    print(f"Datos guardados en {output_file}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()
