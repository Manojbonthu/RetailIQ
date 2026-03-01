# RetailIQ — AI Business Intelligence System

An end-to-end AI-powered business intelligence platform for retail analytics, built with Flask, Scikit-learn, and Claude AI (Anthropic).

---

## 📁 Project Structure

```
retail_ai/
├── app.py                    # Flask application (main entry point)
├── requirements.txt          # Python dependencies
├── data/
│   └── Online_Retail.xlsx    # ← Place your dataset here
├── models/                   # Trained ML models (auto-created)
├── templates/
│   └── index.html            # Full dashboard UI
└── utils/
    ├── __init__.py
    ├── data_processor.py     # Data cleaning, KPIs, RFM analysis
    ├── ml_models.py          # ML pipeline (GBM forecast + K-Means)
    └── chatbot.py            # Claude AI chatbot integration
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
```bash
cp Online_Retail.xlsx data/Online_Retail.xlsx
```

### 3. Run the application
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

### 5. Enable AI Chat
- Enter your Anthropic API key in the top bar (🔑 field)
- Get a free API key at: https://console.anthropic.com
- Click **SAVE** — the key is stored in your browser locally

---

## 🤖 AI Features

| Feature | Description |
|---------|-------------|
| **Revenue Forecasting** | Gradient Boosting Regressor with 16 engineered features (lags, rolling stats, calendar features) |
| **Customer Segmentation** | K-Means clustering + RFM scoring (Champions, Loyal, Potential, At Risk) |
| **AI Chatbot** | Claude Opus (claude-opus-4-6) with full business context — ask anything about your data |
| **Auto Insights** | One-click AI-generated strategic recommendations |

---

## 📊 Dashboard Pages

1. **Dashboard** — KPI overview + revenue trends + geographic + time patterns
2. **Revenue Analysis** — Daily revenue + product revenue + RFM segments
3. **Customers** — Customer segments + country breakdown
4. **Products** — Top products by revenue
5. **Revenue Forecast** — 30-day GBM forecast + feature importance + model metrics
6. **ML Segments** — K-Means cluster analysis + cluster statistics table
7. **AI Chat** — Conversational analytics with Claude AI + quick questions + auto insights

---

## 🔧 Technologies

| Category | Technology |
|----------|-----------|
| **Backend** | Python, Flask |
| **ML** | Scikit-learn (GradientBoostingRegressor, KMeans), Joblib |
| **Data** | Pandas, NumPy |
| **LLM** | Anthropic Claude Opus (claude-opus-4-6) via REST API |
| **Frontend** | HTML5, CSS3, Chart.js, Vanilla JS |
| **Fonts** | Syne (display), Inter (body), JetBrains Mono (code) |

---

## 📈 ML Model Details

### Revenue Forecasting (GradientBoostingRegressor)
- **Features**: Day-of-week, month, lag-1/7/14/30 day revenue, 7/14/30-day rolling averages/std, order counts
- **Training**: 80% historical data, 20% held-out test
- **Evaluation**: R², MAE, RMSE, MAPE reported in dashboard
- **Forecast**: 30-day ahead rolling predictions

### Customer Segmentation (K-Means + RFM)
- **RFM scoring**: Recency, Frequency, Monetary value per customer
- **K-Means**: k=4 clusters with MinMaxScaler normalization
- **Segments**: Champions, Loyal, Potential, At Risk
- **Output**: Per-cluster averages + customer count

---

## ⚡ Performance Notes

- First startup takes 3–5 minutes to train ML models on 541K records
- Models are cached in `models/` — subsequent startups load instantly
- Delete `models/results_cache.pkl` to retrain

---

## 🔑 API Key Security

- The API key is stored only in your browser's `localStorage`
- It is sent directly to Anthropic's servers from your backend
- Never committed to code or logs
