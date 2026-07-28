"""
PASTE THIS AS A NEW CELL AT THE END OF YOUR COLAB NOTEBOOK.

It turns the objects living in your notebook's memory into a self-contained
"model bundle" that the Flask app can load without any notebook code.

It expects these names to already exist (they do, after a full Run All):
    cal_sigmoid or trained   the fitted estimator you want to deploy
    team_features            per (event_code, team) feature table
    X                        the training design matrix (for column order)
    meta                     per-row event_code and red_win
    numeric_feature_cols()   the per-team column list
    SEASON, REGION, PRIMARY_COMPONENT

Output (zipped so you can download it in one click):
    model_bundle/win_model.pkl
    model_bundle/team_features.parquet
    model_bundle/feature_columns.json
    model_bundle/metadata.json
"""
import json
import pickle
import shutil
import sys
from datetime import date
from pathlib import Path

import sklearn

BUNDLE = Path("model_bundle")
BUNDLE.mkdir(exist_ok=True)

# 1) Choose the estimator to deploy. Prefer the calibrated one: it produces the
#    better probabilities, and this app ranks partners BY probability.
model_to_deploy = cal_sigmoid if "cal_sigmoid" in dir() else trained
model_name = type(model_to_deploy).__name__

# 2) The per-team columns the alliance vector is built from. The app cannot call
#    numeric_feature_cols() (it depends on the events_oh global), so store the list.
base_feature_columns = list(numeric_feature_cols())

# 3) Keep only what the app needs: the base feature columns plus the annotation
#    columns the results table displays.
annotation_cols = ["event_code", "team", "qual_rank",
                   "has_color_sorter", "base_capable", "strong_auto", "clean_robot"]
keep = [c for c in dict.fromkeys(annotation_cols + base_feature_columns)
        if c in team_features.columns]
tf_export = team_features[keep].copy()

# Parquet needs clean dtypes. Booleans and nullable ints trip up some readers.
for c in tf_export.columns:
    if str(tf_export[c].dtype) == "boolean" or tf_export[c].dtype == bool:
        tf_export[c] = tf_export[c].astype("float64")
    elif str(tf_export[c].dtype).startswith("Int"):
        tf_export[c] = tf_export[c].astype("float64")
tf_export["team"] = tf_export["team"].astype("int64")

# 4) Held-out metrics, recorded so the app can be honest about what it is showing.
try:
    m = evaluate(model_to_deploy, X[te], meta["red_win"][te])
    metrics = {k: round(float(v), 3) for k, v in m.items() if k != "n"}
    metrics["n_test"] = int(m["n"])
except Exception as exc:                      # never block the export on this
    metrics = {"note": f"metrics unavailable: {exc}"}

metadata = {
    "created": str(date.today()),
    "season": int(SEASON),
    "region": REGION,
    "model_name": model_name,
    "primary_component": PRIMARY_COMPONENT,
    "base_feature_columns": base_feature_columns,
    "n_training_examples": int(len(X)),
    "n_events": int(team_features["event_code"].nunique()),
    "metrics": metrics,
    "sklearn_version": sklearn.__version__,   # MUST match requirements.txt
    "python_version": sys.version.split()[0],
    "caveat": ("Trained on a small sample of playoff matches. Win probability is "
               "measured against a typical alliance at the same event, not a "
               "specific opponent. Use as a ranking aid, not a guarantee."),
}

# 5) Write the four files.
with open(BUNDLE / "win_model.pkl", "wb") as fh:
    pickle.dump(model_to_deploy, fh)
tf_export.to_parquet(BUNDLE / "team_features.parquet", index=False)
(BUNDLE / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=1))
(BUNDLE / "metadata.json").write_text(json.dumps(metadata, indent=1))

# 6) Zip it for a one-click download.
shutil.make_archive("model_bundle", "zip", BUNDLE)

print("Bundle written to model_bundle/ and model_bundle.zip")
print(f"  model            : {model_name}")
print(f"  scikit-learn     : {sklearn.__version__}  <- pin this in requirements.txt")
print(f"  team_features    : {tf_export.shape[0]} rows x {tf_export.shape[1]} cols")
print(f"  feature_columns  : {len(X.columns)}")
print(f"  events           : {metadata['n_events']}")
print(f"  metrics          : {metrics}")

# In Colab, download it:
#   from google.colab import files; files.download("model_bundle.zip")
