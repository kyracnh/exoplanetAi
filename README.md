# ExoplanetAI

Live demo: https://exoplanetai.onrender.com

Required CSV header:
```
koi_score,koi_depth,koi_model_snr,koi_period,koi_duration,koi_prad,koi_srad,koi_kepmag,koi_teq
```

Example row:
```
1.0,1288.3,87.2,9.27358173,3.2875,2.47,0.696,15.302,649.0
```
---

## Live demo

Try the deployed app on Render:

https://exoplanetai.onrender.com

Note: the model may take a short time to finish loading after the web process starts (the site will show the model status badge). If the model is still initializing, wait a minute and reload — uploads/predictions will work once the badge shows "MODEL READY".

---

## Repo layout

- `app.py` — Flask web application (upload page, results page, download endpoint).
- `data_loader.py` — helper to extract and save the features used for training.
- `model_trainer.py` — training pipeline: loads preprocessed CSV, imputes missing values (median), trains RandomForest, evaluates and saves artifacts.
- `model_predictor.py` — loads saved artifacts and makes predictions for new CSVs or single inputs.
- `templates/` — HTML templates for the web UI.
- `models/` — saved model artifacts (model, label encoder, metadata).
- `preprocessed/` — where `preprocessed.csv` is saved by `data_loader.py`.
- `test_data/` — small example CSVs for testing.
- `Dockerfile`, `Makefile`, `requirements.txt` — dev and deployment helpers.

---

## Test using the Live demo (recommended)

Please test the app using the live deployment instead of running local test servers. Open the Live demo URL, upload a CSV like `test_data/test.csv`, and view results in the browser:

https://exoplanetai.onrender.com

Note: the model may need a short warm-up after the web process starts — wait until the status badge shows "MODEL READY" before uploading.

---

## Sample explanation (one input row)

Use this example row (from `test_data/test.csv`) as a minimal CSV with the expected columns:

CSV header:
```
koi_score,koi_depth,koi_model_snr,koi_period,koi_duration,koi_prad,koi_srad,koi_kepmag,koi_teq
```

Example row:
```
1.0,1288.3,87.2,9.27358173,3.2875,2.47,0.696,15.302,649.0
```

What each field means (short):
- koi_score: system score / detection confidence value (0-1)
- koi_depth: transit depth (ppm)
- koi_model_snr: model signal-to-noise ratio
- koi_period: orbital period (days)
- koi_duration: transit duration (hours)
- koi_prad: planet radius (Earth radii)
- koi_srad: stellar radius (Solar radii)
- koi_kepmag: Kepler magnitude (brightness)
- koi_teq: equilibrium temperature (K)

What the app returns and how to read it:
- Prediction label (CONFIRMED / CANDIDATE / FALSE) — the model's class for each row.
- Confidence (%) — the ensemble's top class probability (higher = more confident).
- Class probabilities — per-class probabilities (sum to 1). The UI shows a confidence bar and a numeric percent.
- Result cards — each planet is shown as a card with the prediction badge (color indicates result), confidence, and key feature values.

Example interpretation: if the model returns `CANDIDATE` with 85% confidence, that means the Random Forest ensemble assigned ~0.85 probability to `CANDIDATE` for that input and the other classes lower probabilities (e.g., CONFIRMED 0.10, FALSE 0.05).

If you want a more detailed trace (per-tree votes or model internals), I can add a developer-only inspect page or a helper script (`inspect_tree_votes.py`) that prints per-tree votes for a single input.

## How the model makes predictions (short)

- `model_trainer.py`:
	- Loads `preprocessed/preprocessed.csv`.
	- Separates X and y, imputes missing numeric values with column medians (X = X.fillna(X.median())).
	- Trains a RandomForestClassifier and saves the model and label encoder in `models/`.

- `model_predictor.py`:
	- Loads the saved `random_forest_model.pkl`, `label_encoder.pkl`, and `feature_columns.json`.
	- For incoming CSVs, it ensures the required features are present, fills missing values with the batch median, and calls `model.predict` and `model.predict_proba` to compute class and confidence.

Important: for production consistency you should replace ad-hoc fillna with a scikit-learn `SimpleImputer` fitted at training time and saved with the model (the repo currently uses per-batch median at inference which can lead to drift).

---

## API

- `/` — Upload page (web UI)
- `/upload` (POST) — Upload CSV to get predictions via the UI
- `/download/<filename>` — Download the prediction CSV
- `/api/predict` (POST JSON) — Return prediction(s) for JSON body (single dict or list)

Example JSON payload for `/api/predict`:

```json
{
	"koi_score": 1.0,
	"koi_depth": 1288.3,
	"koi_model_snr": 87.2,
	"koi_period": 9.27358173,
	"koi_duration": 3.2875,
	"koi_prad": 2.47,
	"koi_srad": 0.696,
	"koi_kepmag": 15.302,
	"koi_teq": 649.0
}
```

Response format (example):

```json
{
	"prediction": "CANDIDATE",
	"probabilities": {"CANDIDATE": 0.85, "CONFIRMED": 0.10, "FALSE": 0.05},
	"confidence": 0.85
}
```
