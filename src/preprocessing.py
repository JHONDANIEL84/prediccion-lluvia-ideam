import pandas as pd
import numpy as np
import os

def preprocess_data(input_file='data/rain_data.csv', output_file='data/processed_rain_data.csv'):
    """
    Carga los datos raw, realiza limpieza e ingeniería de características.
    """
    print(f"Cargando datos desde {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convertir fecha a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Ordenar por fecha
    df = df.sort_values('fecha')
    
    # Ingeniería de características de fecha
    df['mes'] = df['fecha'].dt.month
    df['dia_anio'] = df['fecha'].dt.dayofyear
    
    # Lags (Ventanas de tiempo)
    # Usar datos de los últimos 3 días para predecir hoy
    for lag in [1, 2, 3]:
        df[f'precipitacion_lag{lag}'] = df['precipitacion'].shift(lag)
        df[f'temperatura_lag{lag}'] = df['temperatura'].shift(lag)
        df[f'humedad_lag{lag}'] = df['humedad'].shift(lag)
        df[f'presion_lag{lag}'] = df['presion'].shift(lag)
    
    # Rolling stats (Media móvil de 7 días previas)
    df['precipitacion_roll_mean_7'] = df['precipitacion'].shift(1).rolling(window=7).mean()
    
    # Eliminar filas con NaN (generados por los shifts)
    original_len = len(df)
    df = df.dropna()
    print(f"Filas eliminadas por NaN: {original_len - len(df)}")
    
    # Guardar
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Datos preprocesados guardados en {output_file}")
    print(df.head())
    
    return df

if __name__ == "__main__":
    preprocess_data()
