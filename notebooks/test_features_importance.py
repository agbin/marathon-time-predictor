#!/usr/bin/env python3
"""
🧪 TEST: Czy tempo 10km rzeczywiście poprawia dokładność modelu?
Porównanie modeli z 3 vs 4 features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Wczytaj i przygotuj dane jak w głównym skrypcie"""
    print("📊 Wczytywanie danych...")
    
    # Wczytaj dane z obu lat
    df_2023 = pd.read_csv('../data/halfmarathon_wroclaw_2023__final.csv', sep=';')
    df_2024 = pd.read_csv('../data/halfmarathon_wroclaw_2024__final(2).csv', sep=';')
    
    def prepare_year_data(df, year):
        df = df.copy()
        df['age'] = year - df['Rocznik']
        df['finish_time_seconds'] = pd.to_timedelta(df['Czas']).dt.total_seconds()
        df['pace_5km_seconds'] = pd.to_timedelta(df['5 km Tempo']).dt.total_seconds()
        df['pace_10km_seconds'] = pd.to_timedelta(df['10 km Tempo']).dt.total_seconds()
        df['gender_encoded'] = df['Płeć'].map({'M': 1, 'K': 0})
        return df
    
    df_2023 = prepare_year_data(df_2023, 2023)
    df_2024 = prepare_year_data(df_2024, 2024)
    
    # Połącz dane
    df_combined = pd.concat([df_2023, df_2024], ignore_index=True)
    
    # Filtruj dane
    df_ml = df_combined.dropna(subset=['age', 'gender_encoded', 'pace_5km_seconds', 'finish_time_seconds'])
    
    # Usuń outliers
    df_ml = df_ml[
        (df_ml['age'] >= 17) & (df_ml['age'] <= 80) &
        (df_ml['finish_time_seconds'] >= 3600) & (df_ml['finish_time_seconds'] <= 4*3600) &
        (df_ml['pace_5km_seconds'] >= 180) & (df_ml['pace_5km_seconds'] <= 600)
    ]
    
    print(f"✅ Przygotowane dane: {len(df_ml)} rekordów")
    return df_ml

def test_model_comparison():
    """Porównaj modele z 3 vs 4 features"""
    df_ml = load_and_prepare_data()
    
    # Przygotuj target
    y = df_ml['finish_time_seconds']
    
    # MODEL 1: Tylko 3 features (bez tempo 10km)
    X_3features = df_ml[['age', 'gender_encoded', 'pace_5km_seconds']]
    
    # MODEL 2: 4 features (z tempem 10km)
    # Użyj tylko rekordów które mają tempo 10km
    df_with_10km = df_ml.dropna(subset=['pace_10km_seconds'])
    X_4features = df_with_10km[['age', 'gender_encoded', 'pace_5km_seconds', 'pace_10km_seconds']]
    y_4features = df_with_10km['finish_time_seconds']
    
    print(f"\n🔬 PORÓWNANIE MODELI:")
    print(f"📊 Model 3-features: {len(X_3features)} rekordów")
    print(f"📊 Model 4-features: {len(X_4features)} rekordów")
    
    # Test model 3-features
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
        X_3features, y, test_size=0.2, random_state=42
    )
    
    model_3 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_3.fit(X_train_3, y_train_3)
    pred_3 = model_3.predict(X_test_3)
    
    mae_3 = mean_absolute_error(y_test_3, pred_3) / 60  # w minutach
    r2_3 = r2_score(y_test_3, pred_3)
    
    # Test model 4-features
    X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(
        X_4features, y_4features, test_size=0.2, random_state=42
    )
    
    model_4 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_4.fit(X_train_4, y_train_4)
    pred_4 = model_4.predict(X_test_4)
    
    mae_4 = mean_absolute_error(y_test_4, pred_4) / 60  # w minutach
    r2_4 = r2_score(y_test_4, pred_4)
    
    # Wyniki
    print(f"\n🏆 WYNIKI PORÓWNANIA:")
    print(f"📊 MODEL 3-FEATURES (wiek, płeć, tempo_5km):")
    print(f"   ├─ MAE: {mae_3:.1f} minut")
    print(f"   └─ R²: {r2_3:.3f}")
    
    print(f"📊 MODEL 4-FEATURES (+ tempo_10km):")
    print(f"   ├─ MAE: {mae_4:.1f} minut")
    print(f"   └─ R²: {r2_4:.3f}")
    
    improvement_mae = mae_3 - mae_4
    improvement_r2 = r2_4 - r2_3
    
    print(f"\n🎯 POPRAWA DZIĘKI TEMPO 10KM:")
    print(f"   ├─ MAE lepsze o: {improvement_mae:.1f} minut")
    print(f"   └─ R² lepsze o: {improvement_r2:.3f}")
    
    if improvement_mae > 0:
        print(f"✅ WNIOSEK: Tempo 10km POPRAWIA dokładność o {improvement_mae:.1f} min!")
    else:
        print(f"❌ WNIOSEK: Tempo 10km nie poprawia dokładności")
    
    # Feature importance
    print(f"\n🔍 WAŻNOŚĆ CECH W MODELU 4-FEATURES:")
    feature_names = ['age', 'gender_encoded', 'pace_5km_seconds', 'pace_10km_seconds']
    importances = model_4.feature_importances_
    
    for name, importance in zip(feature_names, importances):
        print(f"   ├─ {name}: {importance:.3f} ({importance*100:.1f}%)")

if __name__ == "__main__":
    test_model_comparison()
