# ExoplanetAI

Website: https://exoplanetai.onrender.com/

Required CSV header:
```
koi_score,koi_depth,koi_model_snr,koi_period,koi_duration,koi_prad,koi_srad,koi_kepmag,koi_teq
```

Example row:
```
1.0,1288.3,87.2,9.27358173,3.2875,2.47,0.696,15.302,649.0
```

## How it works (simple)

- Train once: `model_trainer.py` learns from past labeled data and saves the model in `models/`.
- Webapp loads the saved model (now in background) and serves the upload UI.
- When you upload a CSV the app:
	1. Checks the required columns are present.
	2. Fills any missing numbers with column medians.
	3. Sends rows to the RandomForest model which combines many decision trees.
- The app shows:
	- Prediction label (CONFIRMED / CANDIDATE / FALSE)
	- Confidence (top probability, e.g. 96.7%)
	- Per-class probabilities (they add up to 100%)

Example output for the sample row above:
```
Prediction: CANDIDATE
Confidence: 96%
Probabilities: {CANDIDATE: 0.96, CONFIRMED: 0.03, FALSE: 0.01}
```

If you want a developer view (per-tree votes) I can add a small script or debug page.

Output: Prediction label, Confidence (%), per-class probabilities.
