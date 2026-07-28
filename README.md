# FTC Alliance Forecaster

Predicting which partner forms the strongest alliance in FIRST Tech Challenge, and
serving those recommendations through a small web app.

**Stakeholder:** FTC Team 30682, NanoGurus
**Course:** DATA 975 Capstone, Drew University
**Data:** FTCScout public API (DECODE 2025 to 2026 season), cross validated against
The Orange Alliance

---

## What this repository contains

| Path | What it is |
|---|---|
| `notebooks/` | The Colab notebook: collection, cleaning, EDA, features, modeling |
| `colab/export_bundle.py` | Cell to paste into Colab that exports the deployable model bundle |
| `colab/make_demo_bundle.py` | Builds a synthetic bundle so the app runs before the real export |
| `app/` | Flask application (routes, inference layer, templates, CSS) |
| `model_bundle/` | The exported model and feature tables the app loads |
| `tests/` | Tests for the app |
| `wsgi.py` | Entry point for local runs and production servers |

---

## The idea in one paragraph

Public FTC data records scores per **alliance** (two robots combined), never per
robot. To rate a single robot we estimate its individual contribution from many
alliance level results using **OPR**, computed by least squares over qualification
matches. Those per robot ratings become alliance level features, playoff results
become labels, and a calibrated classifier predicts the probability one alliance
beats another. The web app turns that into the question a captain actually asks:
given who is still available, who should we pick?

---

## Running the app locally

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/ftc-alliance-forecaster.git
cd ftc-alliance-forecaster

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Get a model bundle. Either export the real one from the notebook
#    (see colab/export_bundle.py) and unzip it into model_bundle/,
#    or build the synthetic demo bundle:
python colab/make_demo_bundle.py

# 5. Run
python wsgi.py
```

Open http://127.0.0.1:5000

---

## Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Form: choose event and captain |
| `/recommend` | POST | Ranked partner table (HTML) |
| `/teams?event=CODE` | GET | Teams at an event (JSON, fills the dropdown) |
| `/api/recommend?event=CODE&captain=N` | GET | Ranked partners (JSON) |
| `/health` | GET | Liveness check and whether a model is loaded |

Example:

```bash
curl "http://127.0.0.1:5000/api/recommend?event=USNJUCLT2&captain=30682&top_n=5"
```

---

## The model bundle

The app never imports notebook code. The notebook exports four files and the app
reads them:

| File | Contents |
|---|---|
| `win_model.pkl` | The fitted, calibrated scikit-learn estimator |
| `team_features.parquet` | One row per (event_code, team) with the model features |
| `feature_columns.json` | Exact training column order, so inference matches training |
| `metadata.json` | Season, region, metrics, library versions, caveats |

**Version pinning matters.** A scikit-learn pickle should be loaded by the same
version that created it. `metadata.json` records the training version; keep
`requirements.txt` in agreement with it.

---

## Tests

```bash
pip install pytest
pytest -q
```

---

## Honest limitations

- The model is trained on a small number of playoff matches. Treat metrics as
  provisional and widen the region or season count to strengthen them.
- OPR is an estimate of a robot's contribution, weakest for teams with few matches.
- Win probability is measured against a **typical** alliance at the same event, not
  against a specific opponent.
- Playoffs are single elimination and high variance. This is a ranking aid, not a
  guarantee.

---

*Data courtesy of FTCScout and FIRST Tech Challenge.*
