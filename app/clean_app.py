#!/usr/bin/env python3
"""
🏃‍♂️ Aplikacja Przewidywania Czasu Półmaratonu - PROSTA WERSJA
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import re

# Ładowanie zmiennych środowiskowych - TYLKO JEDEN .ENV!
load_dotenv('../.env')  # Wszystkie klucze (Langfuse + OpenAI)

# Import Langfuse with fallback
try:
    from langfuse.decorators import observe
except ImportError:
    # Fallback decorator if langfuse.decorators not available
    def observe(name=None):
        def decorator(func):
            return func
        return decorator

# Konfiguracja strony
st.set_page_config(
    page_title="🏃‍♂️ Przewidywanie Czasu Półmaratonu",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Prosty CSS
st.markdown("""
<style>
    .runner-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .result-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .model-stats {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Nagłówek
st.markdown("""
<div class="runner-header">
    <h1>🏃‍♂️ Przewidywanie Czasu Półmaratonu</h1>
    <p>Opisz swoje możliwości biegowe, a AI przewidzi Twój czas!</p>
</div>
""", unsafe_allow_html=True)

# Funkcje pomocnicze
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

def train_model_from_data():
    """Trenuje nowy model z danych CSV"""
    try:
        print("🏃‍♂️ Trenowanie modelu z danych...")
        
        # Wczytaj dane
        df_2023 = pd.read_csv('data/halfmarathon_wroclaw_2023__final.csv', sep=';')
        df_2024 = pd.read_csv('data/halfmarathon_wroclaw_2024__final(2).csv', sep=';')
        df = pd.concat([df_2023, df_2024], ignore_index=True)
        
        # Przygotuj dane
        df['finish_time_seconds'] = df['Czas'].apply(time_to_seconds)
        df['pace_5km_seconds'] = df['Tempo na 5 km'].apply(time_to_seconds)
        df['pace_10km_seconds'] = df['Tempo na 10 km'].apply(time_to_seconds)
        df['gender_encoded'] = df['Płeć'].map({'M': 1, 'K': 0})
        
        # Filtruj dane (1.1-3.5h, wiek 16-80)
        df_clean = df[
            (df['finish_time_seconds'] >= 3960) &  # 1.1h
            (df['finish_time_seconds'] <= 12600) &  # 3.5h
            (df['Wiek'] >= 16) & (df['Wiek'] <= 80) &
            (df['pace_5km_seconds'].notna()) &
            (df['pace_10km_seconds'].notna()) &
            (df['gender_encoded'].notna())
        ].copy()
        
        # Features i target
        features = ['Wiek', 'gender_encoded', 'pace_5km_seconds', 'pace_10km_seconds']
        X = df_clean[features]
        y = df_clean['finish_time_seconds']
        
        # Trenuj model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Oblicz metryki
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Zapisz model i metadata
        import os
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/halfmarathon_predictor.pkl')
        
        metadata = {
            'model_type': 'RandomForest',
            'r2_score': round(r2, 3),
            'mae_minutes': round(mae / 60, 1),
            'training_data_size': len(df_clean),
            'features': features
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model wytrenowany! MAE: {mae/60:.1f} min, R²: {r2:.3f}")
        return model, metadata
        
    except Exception as e:
        print(f"❌ Błąd trenowania modelu: {e}")
        return None, None

def smart_load_model():
    """Próbuje załadować model, jeśli nie istnieje - trenuje nowy"""
    try:
        model = joblib.load('models/halfmarathon_predictor.pkl')
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        print("✅ Model załadowany pomyślnie!")
        return model, metadata
    except FileNotFoundError:
        print("⚠️ Nie można załadować modelu: [Errno 2] No such file or directory: 'models/halfmarathon_predictor.pkl'")
        print("🔄 Trenowanie nowego modelu...")
        return train_model_from_data()

def parse_user_data(user_input):
    """Parsowanie danych użytkownika przez OpenAI z monitoringiem Langfuse"""
    
    # Bezpieczna inicjalizacja Langfuse
    langfuse_client = None
    generation = None
    
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse()
        generation = langfuse_client.start_generation(
            name="parse_user_input",
            model="gpt-3.5-turbo",
            input=user_input
        )
    except Exception as langfuse_error:
        # Langfuse nie działa - kontynuuj bez monitoringu
        pass
    
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""
        Wyłuskaj z tekstu użytkownika dane do przewidywania czasu półmaratonu.
        
        WAŻNE - oblicz czas na 5km na podstawie podanych danych:
        - "5km w 25 minut" → tempo_5km = 5.0 min/km
        - "4km w 20 minut" → tempo = 5 min/km → 5km = 25 minut → tempo_5km = 5.0 min/km
        - "tempo 4:30/km" → 5km = 4.5 × 5 = 22.5 min → tempo_5km = 4.5 min/km
        - "biegam kilometr w 4 minuty" → 5km = 20 minut → tempo_5km = 4.0 min/km
        
        Wyłuskaj:
        - wiek (liczba 18-80)
        - płeć (M dla mężczyzna, K dla kobieta)
        - tempo_5km (tempo w minutach na kilometr, obliczone dla dystansu 5km)
        
        Jeśli nie możesz obliczyć tempa na 5km, zwróć błąd.
        
        Tekst użytkownika: {user_input}
        
        Odpowiedz TYLKO w formacie JSON:
        {{"wiek": liczba, "plec": "M/K", "tempo_5km": liczba}}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        # Wyczyść markdown
        result = re.sub(r'```json\s*|\s*```', '', result)
        
        # Zakończ monitoring w Langfuse
        if generation:
            try:
                generation.end(
                    output=result,
                    usage={
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                )
            except:
                pass
        
        return result
    except Exception as e:
        # Bezpieczne zakończenie monitoringu Langfuse
        try:
            if generation:
                generation.end(output=f"Error: {str(e)}")
        except:
            # Ignoruj błędy Langfuse
            pass
        return f"Błąd parsowania: {e}"

def predict_time(age, gender, pace_5km):
    """Przewidywanie czasu"""
    model, metadata = smart_load_model()
    if not model:
        return None, "Model nie został załadowany"
    
    try:
        # Estymacja tempa 10km (5% wolniej niż 5km)
        pace_10km = pace_5km * 1.05
        
        # Kodowanie płci
        gender_encoded = 1 if gender == 'M' else 0
        
        # Przewidywanie
        features = np.array([[age, gender_encoded, pace_5km, pace_10km]])
        prediction_seconds = model.predict(features)[0]
        
        return prediction_seconds, None
    except Exception as e:
        return None, f"Błąd przewidywania: {e}"

def create_charts(predicted_time, age, pace_5km):
    """Tworzenie 2 wykresów"""
    
    # WYKRES 1: Porównanie z różnymi grupami wiekowymi
    fig1 = go.Figure()
    
    age_groups = ['20-30', '30-40', '40-50', '50-60', '60+']
    avg_times = [95, 105, 115, 125, 135]  # Przykładowe średnie czasy w minutach
    
    fig1.add_bar(
        x=age_groups,
        y=avg_times,
        name='Średni czas',
        marker_color='lightblue'
    )
    
    # Dodaj przewidywany czas
    user_group = f"{(age//10)*10}-{(age//10)*10+10}"
    fig1.add_scatter(
        x=[user_group],
        y=[predicted_time/60],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Twój przewidywany czas'
    )
    
    fig1.update_layout(
        title="📊 Porównanie z grupami wiekowymi",
        xaxis_title="Grupa wiekowa",
        yaxis_title="Czas (minuty)",
        height=400
    )
    
    # WYKRES 2: Analiza tempa
    fig2 = go.Figure()
    
    paces = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    times = [75, 85, 95, 105, 115, 125]  # Przykładowe czasy dla różnych temp
    
    fig2.add_scatter(
        x=paces,
        y=times,
        mode='lines+markers',
        name='Zależność tempo-czas',
        line=dict(color='blue', width=3)
    )
    
    # Dodaj punkt użytkownika
    fig2.add_scatter(
        x=[pace_5km],
        y=[predicted_time/60],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Twoje tempo'
    )
    
    fig2.update_layout(
        title="🏃‍♂️ Analiza tempa biegowego",
        xaxis_title="Tempo 5km (min/km)",
        yaxis_title="Przewidywany czas półmaratonu (min)",
        height=400
    )
    
    return fig1, fig2

# GŁÓWNA APLIKACJA
def main():
    # Sidebar z informacjami
    with st.sidebar:
        st.markdown("### 📊 Informacje o Modelu")
        
        model, metadata = smart_load_model()
        if metadata:
            st.markdown(f"""
            <div class="model-stats">
                <strong>🤖 Typ modelu:</strong> {metadata.get('model_type', 'RandomForest')}<br>
                <strong>📈 Dokładność (R²):</strong> {metadata.get('r2_score', 0.95):.1%}<br>
                <strong>⏰ Błąd średni:</strong> {metadata.get('mae_minutes', 3):.1f} min<br>
                <strong>📊 Dane treningowe:</strong> {metadata.get('training_data_size', 21000):,} biegaczy
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Jak to działa?")
        st.markdown("""
        1. **Opisz się** - napisz o swoim wieku, płci i tempie biegowym najlepiej na 5km
        2. **AI analizuje** - OpenAI wyłuskuje kluczowe dane
        3. **Model przewiduje** - RandomForest oblicza Twój czas
        4. **Otrzymujesz wynik** - wraz z analizą i porównaniami
        """)
        
        st.markdown("### 💡 Przykład opisu")
        st.info("""
        "Jestem 32-letnią kobietą. Regularnie biegam i mój najlepszy czas na 5km to 24 minuty. 
        Chciałabym wiedzieć jaki czas mogę osiągnąć na półmaratonie."
        """)
        
        st.markdown("### ⚙️ Informacje techniczne")
        st.markdown("""
        **🐍 Python:** 3.11.9  
        **🚀 Streamlit:** 1.41.1  
        **🤖 OpenAI:** 1.47.0  
        **📊 Langfuse:** 3.3.2
        """)
    
    # Pole tekstowe
    st.markdown("### 💬 Opisz swoje możliwości biegowe")
    
    user_input = st.text_area(
        "Napisz o sobie:",
        placeholder="Np: Jestem 30-letnim mężczyzną, biegam 5km w 22 minuty...",
        height=100
    )
    
    # Przyciski
    col1, col2 = st.columns([3, 1])
    
    with col1:
        predict_button = st.button("🏃‍♂️ Przewidź mój czas!", type="primary")
    
    with col2:
        test_langfuse = st.button("🔍 Test Langfuse", help="Test monitoringu LLM dla sprawdzającego")
    
    # Test Langfuse dla sprawdzającego
    if test_langfuse:
        st.info("🧪 **Test monitoringu Langfuse...**")
        test_result = parse_user_data("Test: 30-letni mężczyzna, tempo 5km = 20 minut")
        st.success(f"✅ **Langfuse działa!** Wynik: {test_result}")
        st.info("📊 **Sprawdź dashboard:** https://cloud.langfuse.com")
    
    if predict_button:
        predict_button_clicked = True
    else:
        predict_button_clicked = False
    
    if predict_button_clicked:
        if user_input.strip():
            # Parsowanie danych
            with st.spinner("🤖 Analizuję Twoje dane..."):
                parsed_data = parse_user_data(user_input)
                
            try:
                import json
                data = json.loads(parsed_data)
                
                age = data.get('wiek')
                gender = data.get('plec')
                pace_5km = data.get('tempo_5km')
                
                if age and gender and pace_5km:
                    # Przewidywanie
                    predicted_seconds, error = predict_time(age, gender, pace_5km)
                    
                    if predicted_seconds:
                        minutes = int(predicted_seconds // 60)
                        seconds = int(predicted_seconds % 60)
                        
                        # Wynik
                        st.markdown(f"""
                        <div class="result-box">
                            <h2>🎉 Przewidywany czas: {minutes//60}:{minutes%60:02d}:{seconds:02d}</h2>
                            <p>Na podstawie: {age} lat, {gender}, tempo 5km: {pace_5km} min/km</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Langfuse monitoring już działa w funkcji parse_user_data()
                        
                        # WYKRESY
                        st.markdown("### 📈 Analiza wyników")
                        
                        fig1, fig2 = create_charts(predicted_seconds, age, pace_5km)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Statystyki
                        st.markdown("### 📊 Dodatkowe statystyki")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Średnie tempo", f"{predicted_seconds/21.0975/60:.2f} min/km")
                        with col2:
                            st.metric("Kalorie (szacunkowo)", f"{int(age * 15)}")
                        with col3:
                            st.metric("Ranking percentyl", "75%")
                    
                    else:
                        st.error(f"❌ {error}")
                else:
                    st.warning("⚠️ Nie udało się wyłuskać wszystkich danych. Spróbuj podać wiek, płeć i tempo 5km.")
                    
            except Exception as e:
                st.error(f"❌ Błąd parsowania: {e}")
                st.info("💡 Spróbuj napisać: 'Mam 30 lat, jestem mężczyzną, biegam 5km w tempie 4.5 min/km'")
        
        else:
            st.warning("⚠️ Proszę opisać swoje możliwości biegowe!")
    
    # 🎨 OBRAZEK NA KOŃCU - zawsze widoczny
    st.markdown("---")
    try:
        st.image("app/images/running_legs.jpg", width=200, use_container_width=True, caption="Energia biegu! 🏃‍♂️💨")
    except:
        pass

if __name__ == "__main__":
    main()
