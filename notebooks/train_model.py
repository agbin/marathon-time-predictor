#!/usr/bin/env python3
"""
🏃‍♂️ Trening Modelu Przewidywania Czasu Półmaratonu
Prosty skrypt Python zamiast notebooka
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def time_to_seconds(time_str):
    """Konwertuje czas w formacie HH:MM:SS na sekundy"""
    if pd.isna(time_str):
        return np.nan
    
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            return np.nan
    except:
        return np.nan

def prepare_halfmarathon_data(df, year):
    """Przygotowuje dane półmaratonu do treningu modelu"""
    print(f"🧹 Przygotowywanie danych {year}...")
    
    df_clean = df.copy()
    
    # 1. Dodaj rok
    df_clean['year'] = year
    
    # 2. Oblicz wiek z rocznika
    if 'Rocznik' in df_clean.columns:
        df_clean['age'] = year - df_clean['Rocznik']
        print(f"   ✅ Obliczono wiek: {df_clean['age'].min()}-{df_clean['age'].max()} lat")
    
    # 3. Konwertuj czas końcowy na sekundy
    if 'Czas' in df_clean.columns:
        df_clean['finish_time_seconds'] = df_clean['Czas'].apply(time_to_seconds)
        print(f"   ✅ Konwersja czasu końcowego na sekundy")
    
    # 4. Konwertuj tempo na sekundy na km
    if '5 km Tempo' in df_clean.columns:
        df_clean['pace_5km_seconds'] = df_clean['5 km Tempo'] * 60  # z minut na sekundy
        print(f"   ✅ Konwersja tempa 5km na sekundy/km")
    
    if '10 km Tempo' in df_clean.columns:
        df_clean['pace_10km_seconds'] = df_clean['10 km Tempo'] * 60  # z minut na sekundy
        print(f"   ✅ Konwersja tempa 10km na sekundy/km")
    
    # 5. Kodowanie płci
    if 'Płeć' in df_clean.columns:
        df_clean['gender_encoded'] = df_clean['Płeć'].map({'M': 1, 'K': 0})
        gender_counts = df_clean['Płeć'].value_counts()
        print(f"   ✅ Kodowanie płci: {dict(gender_counts)}")
    
    return df_clean

def predict_halfmarathon_time(model, feature_columns, age, gender, pace_5km_minutes):
    """Przewiduje czas półmaratonu"""
    # Przygotuj features
    gender_encoded = 1 if gender == 'M' else 0
    pace_5km_seconds = pace_5km_minutes * 60
    
    # Przygotuj dane do predykcji
    if len(feature_columns) == 3:
        features = np.array([[age, gender_encoded, pace_5km_seconds]])
    else:  # jeśli mamy też tempo 10km
        # Oszacuj tempo 10km jako podobne do 5km (może być nieco wolniejsze)
        pace_10km_seconds = pace_5km_seconds * 1.05
        features = np.array([[age, gender_encoded, pace_5km_seconds, pace_10km_seconds]])
    
    # Przewidywanie
    prediction_seconds = model.predict(features)[0]
    
    # Konwersja na czytelny format
    hours = int(prediction_seconds // 3600)
    minutes = int((prediction_seconds % 3600) // 60)
    seconds = int(prediction_seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    print("🏃‍♂️ TRENING MODELU PRZEWIDYWANIA CZASU PÓŁMARATONU")
    print("=" * 60)
    
    # 📊 WCZYTANIE DANYCH
    print("\n📊 Wczytywanie danych z separatorem ';'...")
    
    try:
        df_2023 = pd.read_csv('data/halfmarathon_wroclaw_2023__final.csv', sep=';')
        df_2024 = pd.read_csv('data/halfmarathon_wroclaw_2024__final(2).csv', sep=';')
        
        print(f"📈 Dane 2023: {df_2023.shape[0]} rekordów, {df_2023.shape[1]} kolumn")
        print(f"📈 Dane 2024: {df_2024.shape[0]} rekordów, {df_2024.shape[1]} kolumn")
        
    except Exception as e:
        print(f"❌ Błąd wczytywania danych: {e}")
        return
    
    # 🧹 PRZYGOTOWANIE DANYCH
    df_2023_clean = prepare_halfmarathon_data(df_2023, 2023)
    df_2024_clean = prepare_halfmarathon_data(df_2024, 2024)
    
    # 🔗 POŁĄCZENIE DANYCH
    print("\n🔗 Łączenie danych z obu lat...")
    common_cols = list(set(df_2023_clean.columns) & set(df_2024_clean.columns))
    
    df_combined = pd.concat([
        df_2023_clean[common_cols],
        df_2024_clean[common_cols]
    ], ignore_index=True)
    
    print(f"✅ Połączone dane: {df_combined.shape[0]} rekordów, {df_combined.shape[1]} kolumn")
    print(f"📈 Rozkład lat: {df_combined['year'].value_counts().to_dict()}")
    
    # 🤖 PRZYGOTOWANIE DANYCH DO TRENINGU
    print("\n🤖 Przygotowanie danych do treningu...")
    
    # Usuń rekordy z brakującymi kluczowymi danymi
    required_features = ['age', 'gender_encoded', 'pace_5km_seconds', 'finish_time_seconds']
    df_ml = df_combined.dropna(subset=required_features).copy()
    
    print(f"📊 Dane po usunięciu braków: {df_ml.shape[0]} rekordów")
    
    # Filtruj outliers (sensowne zakresy)
    df_ml = df_ml[
        (df_ml['age'] >= 16) & (df_ml['age'] <= 80) &  # Wiek 16-80
        (df_ml['pace_5km_seconds'] >= 180) & (df_ml['pace_5km_seconds'] <= 600) &  # Tempo 5km: 3-10 min/km
        (df_ml['finish_time_seconds'] >= 3600) & (df_ml['finish_time_seconds'] <= 14400)  # Czas: 1-4h
    ]
    
    print(f"📊 Dane po filtracji outliers: {df_ml.shape[0]} rekordów")
    
    # Przygotuj features i target
    feature_columns = ['age', 'gender_encoded', 'pace_5km_seconds']
    if 'pace_10km_seconds' in df_ml.columns and df_ml['pace_10km_seconds'].notna().sum() > len(df_ml) * 0.5:
        feature_columns.append('pace_10km_seconds')
        print("✅ Dodano tempo 10km do features")
    
    X = df_ml[feature_columns]
    y = df_ml['finish_time_seconds']
    
    print(f"🎯 Features: {feature_columns}")
    print(f"📊 X shape: {X.shape}, y shape: {y.shape}")
    print(f"📈 Target range: {y.min():.0f} - {y.max():.0f} sekund ({y.min()/3600:.1f} - {y.max()/3600:.1f} godzin)")
    
    # 🚂 TRENOWANIE MODELU
    print("\n🚂 Trenowanie modelu...")
    
    # Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # Trenuj tylko RandomForest (LinearRegression ma problem z NaN)
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Trenowanie {name}...")
        
        # Trenuj model
        model.fit(X_train, y_train)
        
        # Przewidywania
        y_pred = model.predict(X_test)
        
        # Metryki
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"   📊 MAE: {mae/60:.1f} minut")
        print(f"   📊 RMSE: {rmse/60:.1f} minut")
        print(f"   📊 R²: {r2:.3f}")
    
    # Wybierz najlepszy model (najniższe MAE)
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Najlepszy model: {best_model_name}")
    print(f"   📊 MAE: {results[best_model_name]['mae']/60:.1f} minut")
    print(f"   📊 R²: {results[best_model_name]['r2']:.3f}")
    
    # 💾 ZAPIS MODELU I METADANYCH
    print("\n💾 Zapisywanie modelu...")
    
    # Zapisz model
    model_path = 'models/halfmarathon_predictor.pkl'
    joblib.dump(best_model, model_path)
    print(f"✅ Model zapisany: {model_path}")
    
    # Zapisz metadane modelu
    model_metadata = {
        'model_type': best_model_name,
        'features': feature_columns,
        'mae_minutes': results[best_model_name]['mae'] / 60,
        'r2_score': results[best_model_name]['r2'],
        'training_data_size': len(X_train),
        'feature_ranges': {
            'age': [int(X['age'].min()), int(X['age'].max())],
            'pace_5km_seconds': [int(X['pace_5km_seconds'].min()), int(X['pace_5km_seconds'].max())]
        }
    }
    
    metadata_path = 'models/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"✅ Metadane zapisane: {metadata_path}")
    
    # 🧪 TEST MODELU
    print("\n🧪 Test modelu na przykładowych danych...")
    
    # Przykładowe dane testowe
    test_cases = [
        {'age': 30, 'gender': 'M', 'pace_5km': 4.0},  # Mężczyzna, 30 lat, tempo 4 min/km
        {'age': 25, 'gender': 'K', 'pace_5km': 5.0},  # Kobieta, 25 lat, tempo 5 min/km
        {'age': 45, 'gender': 'M', 'pace_5km': 3.5},  # Mężczyzna, 45 lat, tempo 3.5 min/km
    ]
    
    print("🏃‍♂️ Przykładowe przewidywania:")
    for i, case in enumerate(test_cases, 1):
        predicted_time = predict_halfmarathon_time(
            best_model, feature_columns, case['age'], case['gender'], case['pace_5km']
        )
        print(f"  {i}. {case['gender']}, {case['age']} lat, tempo {case['pace_5km']} min/km → {predicted_time}")
    
    print("\n🎉 MODEL GOTOWY DO UŻYCIA W APLIKACJI!")
    return best_model, feature_columns, model_metadata

if __name__ == "__main__":
    main()
