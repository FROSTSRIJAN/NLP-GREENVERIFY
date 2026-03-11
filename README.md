# GreenVerify – Climate Risk Prediction API

A production-ready FastAPI backend that serves a trained XGBoost model to predict climate risk levels (Low / Medium / High) based on energy and sustainability indicators.

---

## Project Structure

```
greenverify/
├── backend/
│   └── app.py            # FastAPI application
├── models/
│   ├── green_esg_model.pkl   # Trained XGBoost model
│   └── scaler.pkl            # Fitted StandardScaler
├── requirements.txt
└── README.md
```

---

## API Endpoints

### Health Check

```
GET /
```

Response:
```json
{ "message": "GreenVerify Climate Risk API running" }
```

### Predict Climate Risk

```
POST /predict
Content-Type: application/json
```

Request body:
```json
{
  "gdp": 14722.84,
  "population": 331002651,
  "coal_consumption": 11.58,
  "gas_consumption": 32.29,
  "oil_consumption": 35.36,
  "renewables_consumption": 6.28,
  "solar_consumption": 1.58,
  "wind_consumption": 2.73,
  "hydro_consumption": 2.62
}
```

Response:
```json
{ "risk_level": "Medium Risk" }
```

### Interactive Docs

Swagger UI is available at **`/docs`** once the server is running.

---

## Run Locally

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn backend.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

## Deploy on Render

1. Push this repository to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Connect the GitHub repo.
4. Configure the service:

| Setting           | Value                                              |
| ----------------- | -------------------------------------------------- |
| **Runtime**       | Python                                             |
| **Build Command** | `pip install -r requirements.txt`                  |
| **Start Command** | `uvicorn backend.app:app --host 0.0.0.0 --port $PORT` |

5. Deploy. Render will install dependencies, start the server, and assign a public URL.

---

## Risk Label Mapping

| Prediction Value | Label        |
| ---------------- | ------------ |
| 0                | Low Risk     |
| 1                | Medium Risk  |
| 2                | High Risk    |
