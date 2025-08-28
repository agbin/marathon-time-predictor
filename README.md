# 🏃‍♂️ Aplikacja Przewidywania Czasu Półmaratonu

Aplikacja do przewidywania czasu półmaratonu na podstawie danych użytkownika (wiek, płeć, tempo 5km) z wykorzystaniem Machine Learning i monitoringu LLM.

## 🎯 Funkcjonalności

- **🤖 Parsowanie LLM**: OpenAI GPT-3.5 wyłuskuje dane z naturalnego języka
- **📊 Model ML**: RandomForest przewiduje czas półmaratonu (MAE ~3 min, R²=95%)
- **📈 Wizualizacje**: Interaktywne wykresy Plotly z analizą wieku i tempa
- **🔍 Monitoring**: Langfuse śledzi skuteczność parsowania LLM
- **🎨 UI**: Nowoczesny interfejs Streamlit z responsywnym designem

## 🛠️ Technologie

- **Backend**: Python 3.9+, scikit-learn, pandas, numpy
- **Frontend**: Streamlit, Plotly
- **AI/ML**: OpenAI GPT-3.5-turbo, RandomForest
- **Monitoring**: Langfuse
- **Deployment**: Digital Ocean App Platform

## 📋 Wymagania

### Pakiety Python
```bash
streamlit>=1.28.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
openai>=1.0.0
langfuse>=2.0.0
python-dotenv>=1.0.0
joblib>=1.3.0
```

### Klucze API
- **OpenAI API Key** - do parsowania danych użytkownika
- **Langfuse Keys** - do monitoringu LLM (opcjonalne)

## 🚀 Instalacja i Uruchomienie

### 1. Klonowanie repozytorium
```bash
git clone <repository-url>
cd zadanie_domowe_polmaraton
```

### 2. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 3. Konfiguracja zmiennych środowiskowych

Stwórz plik `.env` w głównym katalogu projektu:

```env
# OpenAI API (WYMAGANE)
OPENAI_API_KEY=sk-proj-your-openai-api-key-here

# Langfuse Monitoring (OPCJONALNE)
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 4. Uruchomienie aplikacji
```bash
cd app
streamlit run clean_app.py
```

Aplikacja będzie dostępna pod adresem: `http://localhost:8501`

## 📁 Struktura projektu

```
zadanie_domowe_polmaraton/
├── .env                    # Zmienne środowiskowe (klucze API)
├── README.md              # Ten plik
├── requirements.txt       # Zależności Python
├── app/
│   ├── clean_app.py      # Główna aplikacja Streamlit
│   └── images/           # Zdjęcia do UI
├── models/
│   ├── halfmarathon_predictor.pkl  # Wytrenowany model ML
│   └── model_metadata.json         # Metadane modelu
├── data/                  # Dane treningowe (opcjonalne)
└── notebooks/            # Jupyter notebooks do trenowania
```

## 🎮 Jak używać

### 1. Wprowadź dane
W polu tekstowym wpisz swoje dane w naturalnym języku, np.:
- "Jestem 30-letnim mężczyzną, biegam 5km w 25 minut"
- "Kobieta, 28 lat, tempo 5:00/km"
- "Mam 35 lat, płeć męska, kilometr w 4 minuty"

### 2. Przewidywanie
Kliknij "🏃‍♂️ Przewidź mój czas!" - aplikacja:
- Wyłuska dane przez OpenAI
- Przewidzi czas półmaratonu
- Pokaże wykresy i analizy

### 3. Test monitoringu (opcjonalnie)
Kliknij "🔍 Test Langfuse" aby przetestować monitoring LLM.

## 🔧 Deployment na Digital Ocean

### 1. App Platform
- Połącz z repozytorium GitHub
- Wybierz Python app
- Ustaw build command: `pip install -r requirements.txt`
- Ustaw run command: `cd app && streamlit run clean_app.py --server.port=$PORT`

### 2. Zmienne środowiskowe
W panelu Digital Ocean dodaj:
- `OPENAI_API_KEY`
- `LANGFUSE_PUBLIC_KEY` (opcjonalne)
- `LANGFUSE_SECRET_KEY` (opcjonalne)
- `LANGFUSE_HOST` (opcjonalne)

## 📊 Model ML

### Dane treningowe
- **Źródło**: Półmaraton Wrocław 2023-2024
- **Rozmiar**: ~22k rekordów
- **Features**: wiek, płeć, tempo 5km, tempo 10km

### Metryki modelu
- **Algorytm**: RandomForest
- **MAE**: ~3 minuty
- **R² Score**: 95.1%
- **RMSE**: ~4.5 minut

## 🔍 Monitoring Langfuse

Aplikacja automatycznie monitoruje:
- **Skuteczność parsowania** danych użytkownika
- **Koszty API** OpenAI (tokeny)
- **Latencję** odpowiedzi LLM
- **Błędy** w parsowaniu

Dashboard: https://cloud.langfuse.com

## 🐛 Rozwiązywanie problemów

### Błąd: "No module named 'streamlit'"
```bash
pip install streamlit
```

### Błąd: "OpenAI API key not found"
Sprawdź czy plik `.env` zawiera poprawny klucz `OPENAI_API_KEY`

### Błąd: "Model file not found"
Upewnij się że plik `models/halfmarathon_predictor.pkl` istnieje

### Aplikacja nie startuje
```bash
cd app
python -c "from dotenv import load_dotenv; load_dotenv('../.env'); import os; print('OpenAI:', 'OK' if os.getenv('OPENAI_API_KEY') else 'BRAK')"
```

## 👨‍💻 Autor

Projekt stworzony w ramach kursu "Od Zera do AI" - Moduł 9: Zadanie Domowe

## 📝 Licencja

MIT License - możesz swobodnie używać i modyfikować kod.
