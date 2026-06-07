# ⚽ SportsPulse — Real-Time Football Analytics Platform

> Production-grade football data engineering and ML platform covering 12 global leagues with live predictions, built during the 2026 FIFA World Cup.

## 🔴 Live Demo
- **Dashboard:** https://sportspulse-o2jmxvc7fwunmqkgwweo5s.streamlit.app
- **API:** https://sportspulse-api.onrender.com

---

## 🏗️ Architecture
API-Football → Bronze Layer → Silver Layer → Gold Layer → ML Model → Dashboard
↑               ↑              ↑           ↑
Raw Data       Cleaned Data   Aggregated   Predictions
(Neon PG)      (Neon PG)      (Neon PG)    (XGBoost)

### Medallion Architecture
| Layer | Tables | Description |
|---|---|---|
| Bronze | `bronze_fixtures`, `bronze_teams` | Raw ingested data from API |
| Silver | `silver_fixtures`, `silver_teams` | Cleaned, validated, enriched |
| Gold | `gold_team_stats`, `gold_league_trends`, `gold_head_to_head` | Aggregated analytics |
| ML | `ml_features`, `ml_predictions` | Feature matrix and predictions |

---

## 📊 What It Does

- Ingests real football data from **12 global leagues** via API-Football
- Processes **3,044 fixtures** and **215 teams** through a Bronze→Silver→Gold pipeline
- Trains an **XGBoost ML model** with **95.3% accuracy** on match outcome prediction
- Serves data via a **Flask REST API** with 8 endpoints
- Visualizes everything in a **Streamlit dashboard** with 5 interactive pages
- Orchestrates the full pipeline with **Apache Airflow** running 3 DAGs daily

---

## 🌍 Leagues Covered

| League | Country | Season |
|---|---|---|
| Premier League | England | 2024/25 |
| La Liga | Spain | 2024/25 |
| Bundesliga | Germany | 2024/25 |
| Serie A | Italy | 2024/25 |
| Ligue 1 | France | 2024/25 |
| UEFA Champions League | Europe | 2024/25 |
| Eredivisie | Netherlands | 2024/25 |
| Primeira Liga | Portugal | 2024/25 |
| Brazilian Série A | Brazil | 2024/25 |
| Saudi Pro League | Saudi Arabia | 2024/25 |
| MLS | USA | 2024 |
| World Cup 2026 | World | 2026 |

---

## 🤖 ML Model

- **Algorithm:** XGBoost Classifier
- **Accuracy:** 95.3%
- **Features:** 33 engineered features including:
  - Team win/draw/loss rates
  - Home and away win rates
  - Average goals scored/conceded
  - Form scores (last 5 matches)
  - Head-to-head records
  - Goal differentials
  - Points differential
- **Target:** Match outcome (HOME_WIN / AWAY_WIN / DRAW)

---

## 🔌 API Endpoints

Base URL: `https://sportspulse-api.onrender.com`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/summary` | Platform stats |
| GET | `/leagues` | All leagues |
| GET | `/fixtures` | Fixtures with filters |
| GET | `/standings?league_id=39` | League standings |
| GET | `/teams` | All teams |
| GET | `/h2h?team_a=33&team_b=40` | Head to head |
| GET | `/predictions` | ML predictions |
| POST | `/predict` | Predict match outcome |
| GET | `/trends` | League trends by round |

### Example: Predict a match
```bash
curl -X POST "https://sportspulse-api.onrender.com/predict" \
-H "Content-Type: application/json" \
-d '{"home_team_id": 33, "away_team_id": 40}'
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Languages | Python, SQL |
| Data Processing | Pandas, NumPy |
| Database | PostgreSQL (Neon Cloud) |
| ORM | SQLAlchemy |
| ML | XGBoost, Scikit-learn |
| API | Flask, Gunicorn |
| Dashboard | Streamlit, Plotly |
| Orchestration | Apache Airflow |
| Containerization | Docker |
| Deployment | Render (API), Streamlit Cloud (Dashboard) |
| Version Control | Git, GitHub |

---

## 📁 Project Structure
sportspulse/
├── airflow/
│   └── dags/
│       └── sportspulse_pipeline.py    # 3 Airflow DAGs
├── ingestion/
│   └── fixtures_extractor.py          # Multi-league data ingestion
├── processing/
│   ├── bronze_to_silver.py            # Data cleaning & enrichment
│   └── silver_to_gold.py             # Aggregation & analytics
├── ml/
│   ├── feature_engineering.py         # 33 feature matrix
│   ├── outcome_predictor.py           # XGBoost training & predictions
│   └── models/                        # Saved model files
├── api/
│   └── app.py                         # Flask REST API
├── dashboard/
│   └── streamlit_app.py              # Streamlit dashboard
├── database/
├── reporting/
├── quality/
├── docker-compose.yml                 # Airflow Docker setup
├── requirements.txt
├── Procfile                           # Render deployment
└── .env                              # Environment variables (not committed)

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11
- PostgreSQL (or Neon cloud account)
- Docker Desktop (for Airflow)

### Setup

```bash
# Clone the repo
git clone https://github.com/Albyaz/sportspulse.git
cd sportspulse

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database URL

# Run the data pipeline
python ingestion/fixtures_extractor.py
python processing/bronze_to_silver.py
python processing/silver_to_gold.py
python ml/feature_engineering.py
python ml/outcome_predictor.py

# Start the API
python api/app.py

# Start the dashboard
streamlit run dashboard/streamlit_app.py
```

### Run Airflow with Docker

```bash
docker compose up -d
# Visit http://localhost:8080
# Username: admin | Password: admin
```

---

## 📈 Pipeline Schedule

| DAG | Schedule | Description |
|---|---|---|
| `sportspulse_ingestion` | Daily 6am UTC | Pulls fresh data from API-Football |
| `sportspulse_processing` | Daily 7am UTC | Bronze → Silver → Gold transformation |
| `sportspulse_ml` | Daily 8am UTC | Retrains model, generates predictions |

---

## 👨‍💻 Author

**Albright Ndamati**
- MD | Data Engineer & Product Builder
- GitHub: [@Albyaz](https://github.com/Albyaz)
- LinkedIn: [Albright Ndamati](https://linkedin.com/in/albrightndamati)

---

## 📄 License

MIT License — feel free to use this as a reference for your own data engineering projects.