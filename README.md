# Consumer Intelligence Platform

An end-to-end, locally-hosted consumer analytics platform combining advanced ML segmentation, predictive modelling, LLM-powered persona generation, and a premium stakeholder dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Consumer Intelligence Platform                   │
├──────────────────────────┬──────────────────────────────────────────┤
│   Frontend (React+Vite)  │          Backend (FastAPI)               │
│   localhost:5173         │          localhost:8000                  │
│                          │                                          │
│  ┌─────────────────────┐ │  ┌────────────────────────────────────┐ │
│  │  Overview Dashboard │ │  │  /generate-data                    │ │
│  │  Segment Explorer   │ │  │  /train-models                     │ │
│  │  Persona Cards      │◄├──│  /segments                         │ │
│  │  Predictions Panel  │ │  │  /personas                         │ │
│  │  Explainability     │ │  │  /predictions                      │ │
│  │  Fairness Audit     │ │  │  /explanations                     │ │
│  │  What-If Simulator  │ │  │  /bias-report                      │ │
│  │  Data Pipeline      │ │  └────────────────────────────────────┘ │
│  └─────────────────────┘ │                   │                     │
│                          │  ┌────────────────▼───────────────────┐ │
│                          │  │           ML Pipeline              │ │
│                          │  │  DataGenerator → FeatureEng        │ │
│                          │  │  → KMeans/GMM → RF/GBM/LR         │ │
│                          │  │  → SHAP → BiasAudit                │ │
│                          │  └────────────────────────────────────┘ │
│                          │                   │                     │
│                          │  ┌────────────────▼───────────────────┐ │
│                          │  │        Ollama (local LLM)          │ │
│                          │  │  localhost:11434  (llama3)         │ │
│                          │  │  · Synthetic text reviews          │ │
│                          │  │  · Persona narrative generation    │ │
│                          │  └────────────────────────────────────┘ │
├──────────────────────────┴──────────────────────────────────────────┤
│  Storage: /data/consumers.csv  ·  /models/*.pkl                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Module | Description |
|--------|-------------|
| Synthetic Data Generator | 100k+ consumer rows — demographics, behavioural, transactional, time-series, NPS, reviews |
| Feature Engineering | RFM scores, engagement index, digital activity, support burden |
| Segmentation Engine | KMeans + Gaussian Mixture Model with PCA 2D visualisation |
| Prediction Engine | Churn (GBM), LTV (RandomForest), Conversion (Logistic Regression) |
| Persona Generator | Ollama-powered human personas with fallback templates |
| Explainability | SHAP TreeExplainer + permutation importance fallback |
| Fairness Audit | 80% adverse impact rule, demographic parity across gender/income/city |
| What-If Simulator | Real-time feature sliders with instant model predictions |
| Premium Dashboard | Dark-mode, custom CSS, no framework dependencies |

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Backend |
| Node.js | 18+ | Frontend |
| Ollama | Latest | Local LLM (optional — platform works without it) |

---

## Setup Instructions

### 1. Install Ollama (optional but recommended)

Download from https://ollama.com and install.

Pull the llama3 model:
```bash
ollama pull llama3
```

Verify it's running:
```bash
curl http://localhost:11434/api/tags
```

---

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
copy .env.example .env       # Windows
cp .env.example .env         # macOS/Linux
```

Edit `.env` if needed (default values work out of the box):
```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
DATASET_SIZE=100000
N_CLUSTERS=6
```

Start the backend:
```bash
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

---

### 3. Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Open: http://localhost:5173

---

## Running the Full Pipeline

Once both servers are running, open the dashboard and navigate to **Data Pipeline**:

**Step 1 — Generate Dataset**
- Click "Generate Dataset"
- Default: 100,000 rows (adjustable 10k–500k)
- Takes ~30–90 seconds depending on hardware
- Ollama generates review text if available; falls back to templates otherwise

**Step 2 — Train Models**
- Click "Train Models"
- Select number of clusters (default: 6) and method (KMeans or GMM)
- Runs: feature engineering → segmentation → churn/LTV/conversion models
- Takes ~2–5 minutes for 100k rows

After training completes, all dashboard sections become active.

---

## API Reference

### Health
```
GET  /health
```

### Data Generation
```
POST /generate-data?n_rows=100000
GET  /generate-data/status
GET  /generate-data/info
```

### Model Training
```
POST /train-models?n_clusters=6&method=kmeans
GET  /train-models/status
```

### Segmentation
```
GET  /segments
GET  /segments/{segment_id}
```

### Personas
```
GET  /personas
```

### Predictions
```
GET  /predictions
POST /predictions/what-if
     Body: { "features": { "rfm_recency": 0.7, "engagement_score": 0.4, ... } }
```

### Explainability
```
GET  /explanations
```

### Fairness
```
GET  /bias-report
```

### Sample API Calls (curl)

```bash
# Generate 50,000 rows
curl -X POST "http://localhost:8000/generate-data?n_rows=50000"

# Check generation status
curl http://localhost:8000/generate-data/status

# Train models with 5 clusters
curl -X POST "http://localhost:8000/train-models?n_clusters=5&method=kmeans"

# Get segment data
curl http://localhost:8000/segments | python -m json.tool

# What-if prediction
curl -X POST http://localhost:8000/predictions/what-if \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "rfm_recency": 0.2,
      "rfm_frequency": 1.5,
      "engagement_score": 0.1,
      "nps_score": 3,
      "last_purchase_days_ago": 280,
      "support_tickets": 8,
      "email_open_rate": 0.05
    }
  }'

# Bias report
curl http://localhost:8000/bias-report | python -m json.tool
```

---

## Dataset Schema

The generated `data/consumers.csv` contains these columns:

| Column | Type | Description |
|--------|------|-------------|
| customer_id | string | Unique identifier (C0000001…) |
| age | int | 18–80 |
| gender | string | M / F / Non-binary |
| income_bracket | string | low / medium / high / very_high |
| education_level | string | high_school → phd |
| occupation | string | 10 categories |
| city_tier | int | 1 (metro) / 2 / 3 (rural) |
| subscription_type | string | free / basic / premium / enterprise |
| preferred_device | string | mobile / desktop / tablet |
| product_category | string | Top product category |
| account_age_days | int | Days since account creation |
| total_purchases | int | Total purchase count |
| total_spend | float | Cumulative spend ($) |
| avg_order_value | float | Mean order value ($) |
| purchase_frequency_monthly | float | Average monthly purchase rate |
| last_purchase_days_ago | int | Recency signal |
| session_count_30d | int | Sessions in last 30 days |
| avg_session_duration_min | float | Mean session duration |
| pages_per_session | float | Mean pages viewed per session |
| email_open_rate | float | 0–1 |
| email_click_rate | float | 0–1 |
| app_opens_30d | int | App opens in last 30 days |
| support_tickets | int | Open support tickets |
| nps_score | int | Net Promoter Score (0–10) |
| churn_probability | float | Rule-based ground truth (0–1) |
| is_churned | int | Binary churn label |
| ltv_actual | float | Simulated lifetime value ($) |
| converted | int | Binary conversion label |
| monthly_activity | json string | 12-month session activity |
| review_text | string | Customer review (Ollama or template) |
| created_at | date | Account creation date |

---

## ML Models

| Model | Algorithm | Target | Metric |
|-------|-----------|--------|--------|
| Segmentation | KMeans / GMM | Cluster assignment | Silhouette score |
| Churn | GradientBoostingClassifier | is_churned | Accuracy + AUC |
| LTV | RandomForestRegressor | log(ltv_actual) | R² |
| Conversion | LogisticRegression | converted | Accuracy + AUC |

Feature engineering produces 25+ derived features including RFM scores,
engagement index, digital activity score, support burden, and session quality.

---

## Project Structure

```
Consumer Segmentation/
├── backend/
│   ├── main.py               # FastAPI app, CORS, lifespan
│   ├── config.py             # Settings (pydantic-settings)
│   ├── requirements.txt
│   ├── .env.example
│   ├── routers/              # API route handlers
│   │   ├── data.py           # /generate-data
│   │   ├── ml_models.py      # /train-models
│   │   ├── segments.py       # /segments
│   │   ├── personas.py       # /personas
│   │   ├── predictions.py    # /predictions + /what-if
│   │   ├── explanations.py   # /explanations
│   │   └── bias.py           # /bias-report
│   ├── services/             # Business logic
│   │   ├── data_generator.py
│   │   ├── feature_engineering.py
│   │   ├── segmentation.py
│   │   ├── prediction_engine.py
│   │   ├── persona_generator.py
│   │   ├── explainability.py
│   │   └── bias_fairness.py
│   ├── models/
│   │   └── schemas.py        # Pydantic response models
│   └── utils/
│       ├── logger.py         # Loguru setup
│       └── ollama_client.py  # Async Ollama HTTP client
├── frontend/
│   ├── package.json
│   ├── vite.config.js        # Proxy to backend
│   ├── index.html
│   └── src/
│       ├── App.jsx           # Router + layout
│       ├── main.jsx
│       ├── styles/global.css # Full design system
│       ├── services/api.js   # Axios client
│       ├── hooks/useApi.js   # Data-fetching hooks
│       └── components/
│           ├── Layout/       # Sidebar, Header
│           ├── Overview/     # Executive dashboard
│           ├── Segmentation/ # Cluster explorer
│           ├── Personas/     # Persona cards
│           ├── Predictions/  # Model results
│           ├── Explainability/
│           ├── Bias/         # Fairness audit
│           ├── Simulation/   # What-if tool
│           ├── Pipeline/     # Data pipeline UI
│           └── Common/       # Shared components
├── data/                     # Generated CSV (git-ignored)
├── models/                   # Trained pkl files (git-ignored)
├── docs/
├── .gitignore
└── README.md
```

---

## Troubleshooting

**Backend won't start — "module not found"**
Make sure you activated the virtual environment (`venv\Scripts\activate` on Windows) before running uvicorn.

**SHAP import error**
SHAP has C++ build requirements. If installation fails:
```bash
pip install shap --no-build-isolation
```
The platform automatically falls back to permutation importance if SHAP is unavailable.

**Ollama shows "offline" in sidebar**
This is non-blocking. The platform works without Ollama using template-based personas and static review text. To fix, ensure Ollama is running (`ollama serve`) before starting the backend.

**Data generation hangs**
Check `GET /generate-data/status`. If status is "failed", check `backend/logs/app.log` for the error.

**Frontend shows blank/error after training**
Refresh the page. The frontend polls the backend every 3 seconds during training but a manual refresh after training completes is the most reliable way to load fresh data.

**Windows path errors**
Ensure Python 3.11+ and Node 18+ are on PATH. Run `python --version` and `node --version` to verify.

---

## Performance Notes

- 100k rows generates in ~30s and trains in ~2–5 minutes on a modern laptop
- SHAP explanations are computed on a 500-row sample for speed
- Scatter plot is limited to 3,000 sampled points for render performance
- All ML training happens in a background thread to keep the API responsive

---

## Screenshots

```
[Overview Dashboard]        — KPI cards, segment pie, churn histogram
[Segmentation Explorer]     — PCA scatter plot + cluster detail cards
[Persona Cards]             — LLM-generated profiles with metrics
[Predictions Panel]         — Churn/LTV/conversion distributions + feature importance
[Explainability]            — SHAP bar charts + per-customer waterfall
[Fairness Audit]            — AIR table, demographic distributions, recommendations
[What-If Simulator]         — Feature sliders + real-time prediction output
[Data Pipeline]             — Step-by-step pipeline with live log
```
