"""
Build a SYNTHETIC model bundle so the web app can be run and tested before the
real one is exported from Colab.

The schema matches the notebook exactly, so swapping in the real bundle is a
straight file replacement with no code changes.

Run from the project root:
    python colab/make_demo_bundle.py
"""
import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(7)

SCORE_COMPONENTS = ["autoPoints", "dcPoints", "autoArtifactPoints", "dcArtifactPoints",
                    "dcPatternPoints", "dcBasePoints", "totalPointsNp"]
PRIMARY_COMPONENT = "totalPointsNp"
EVENTS = ["USNJDEMO1", "USNJDEMO2", "USNJUCLT2"]

base_cols = ([f"opr_{c}" for c in SCORE_COMPONENTS]
             + [f"avg_{c}" for c in SCORE_COMPONENTS]
             + ["qual_rank", "consistency", "win_rate", "strength_of_schedule", "recent_form",
                "has_color_sorter", "base_capable", "strong_auto", "clean_robot",
                "avg_true_auto_leave", "avg_true_dc_base"]
             + ["event_type_League", "region_USNJ", "state_NJ"])   # event one-hots

rows = []
for ev in EVENTS:
    n = 20
    teams = rng.choice(np.arange(1000, 40000), size=n, replace=False)
    skill = rng.normal(70, 30, size=n).clip(5, 200)
    order = np.argsort(-skill)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    if ev == "USNJUCLT2":
        # Put a familiar team number at seed 2, a realistic captain slot, so the
        # demo has candidates available below it.
        teams[np.where(ranks == 2)[0][0]] = 30682
    for i, t in enumerate(teams):
        r = {"event_code": ev, "team": int(t), "qual_rank": float(ranks[i])}
        for c in SCORE_COMPONENTS:
            share = {"autoPoints": .25, "dcPoints": .70, "autoArtifactPoints": .15,
                     "dcArtifactPoints": .45, "dcPatternPoints": .08,
                     "dcBasePoints": .10, "totalPointsNp": 1.0}[c]
            r[f"opr_{c}"] = float(skill[i] * share + rng.normal(0, 3))
            r[f"avg_{c}"] = float(skill[i] * share * 2 + rng.normal(0, 5))
        r["consistency"] = float(abs(rng.normal(20, 6)))
        r["win_rate"] = float(np.clip(skill[i] / 160 + rng.normal(0, .08), 0, 1))
        r["strength_of_schedule"] = float(rng.normal(120, 15))
        r["recent_form"] = float(skill[i] + rng.normal(0, 10))
        r["has_color_sorter"] = int(skill[i] > 55)
        r["base_capable"] = int(skill[i] > 45)
        r["strong_auto"] = int(skill[i] > 65)
        r["clean_robot"] = int(rng.random() > .35)
        r["avg_true_auto_leave"] = float(rng.uniform(0, 6))
        r["avg_true_dc_base"] = float(rng.uniform(0, 12))
        r["event_type_League"] = 1.0            # identical for every team at an event
        r["region_USNJ"] = 1.0
        r["state_NJ"] = 1.0
        rows.append(r)

team_features = pd.DataFrame(rows)

# Build the alliance-difference design matrix exactly like build_alliance_dataset().
feature_columns = [f"{p}_{c}" for c in base_cols for p in ("sum", "max")]


def alliance_vec(tf_ev, t1, t2):
    f1, f2 = tf_ev.loc[t1], tf_ev.loc[t2]
    v = {}
    for c in base_cols:
        a, b = float(f1[c]), float(f2[c])
        v[f"sum_{c}"] = a + b
        v[f"max_{c}"] = max(a, b)
    return v


X_rows, y = [], []
for ev in EVENTS:
    tf_ev = team_features[team_features["event_code"] == ev].set_index("team")
    ids = list(tf_ev.index)
    for _ in range(60):
        a, b, c, d = rng.choice(ids, size=4, replace=False)
        red, blue = alliance_vec(tf_ev, a, b), alliance_vec(tf_ev, c, d)
        X_rows.append({k: red[k] - blue[k] for k in red})
        strength = (red[f"sum_opr_{PRIMARY_COMPONENT}"] - blue[f"sum_opr_{PRIMARY_COMPONENT}"])
        y.append(int(strength + rng.normal(0, 25) > 0))

X = pd.DataFrame(X_rows)[feature_columns].fillna(0.0)
y = np.array(y)

model = CalibratedClassifierCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
    method="sigmoid", cv=3).fit(X, y)

BUNDLE = Path(__file__).resolve().parent.parent / "model_bundle"
BUNDLE.mkdir(exist_ok=True)
with open(BUNDLE / "win_model.pkl", "wb") as fh:
    pickle.dump(model, fh)
team_features.to_parquet(BUNDLE / "team_features.parquet", index=False)
(BUNDLE / "feature_columns.json").write_text(json.dumps(feature_columns, indent=1))
(BUNDLE / "metadata.json").write_text(json.dumps({
    "created": str(date.today()), "season": 2025, "region": "USNJ (SYNTHETIC DEMO)",
    "model_name": "CalibratedClassifierCV", "primary_component": PRIMARY_COMPONENT,
    "base_feature_columns": base_cols, "n_training_examples": int(len(X)),
    "n_events": len(EVENTS),
    "metrics": {"note": "synthetic demo data, metrics not meaningful"},
    "sklearn_version": sklearn.__version__,
    "caveat": ("DEMO BUNDLE built from synthetic data so the app can be tested. "
               "Replace with the real bundle exported from the notebook."),
}, indent=1))

print(f"Demo bundle written to {BUNDLE}")
print(f"  team_features  : {team_features.shape}")
print(f"  feature_columns: {len(feature_columns)}")
print(f"  events         : {EVENTS}")
