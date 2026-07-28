"""
Tests for the FTC Alliance Forecaster web app.

Run from the project root:
    pip install pytest
    pytest -q

These assume a bundle exists in model_bundle/. If you have not exported the real
one yet, build the demo bundle first:
    python colab/make_demo_bundle.py
"""
import json
from pathlib import Path

import pytest

from app import create_app

BUNDLE = Path(__file__).resolve().parent.parent / "model_bundle"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _first_event(client):
    return client.get("/health").get_json() and \
        json.loads((BUNDLE / "metadata.json").read_text()) is not None


def test_bundle_files_exist():
    for name in ["win_model.pkl", "team_features.parquet",
                 "feature_columns.json", "metadata.json"]:
        assert (BUNDLE / name).exists(), f"missing bundle file: {name}"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
    assert r.get_json()["events_loaded"] > 0


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Recommend partners" in r.data


def test_teams_endpoint(client):
    events = client.get("/health").get_json()["events_loaded"]
    assert events > 0
    # pull an event code straight off the index page options
    page = client.get("/").data.decode()
    code = page.split('<option value="')[2].split('"')[0]
    r = client.get(f"/teams?event={code}")
    assert r.status_code == 200
    assert len(r.get_json()["teams"]) > 0


def test_api_recommend_sorted_desc(client):
    page = client.get("/").data.decode()
    code = page.split('<option value="')[2].split('"')[0]
    r = client.get(f"/api/recommend?event={code}&top_n=5")
    assert r.status_code == 200
    recs = r.get_json()["recommendations"]
    assert len(recs) > 0
    probs = [x["win_prob_vs_field"] for x in recs]
    assert probs == sorted(probs, reverse=True), "recommendations must be ranked best first"
    assert all(0.0 <= p <= 1.0 for p in probs), "probabilities must be in [0, 1]"


def test_unknown_event_returns_400(client):
    r = client.get("/api/recommend?event=DOES_NOT_EXIST")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_unknown_captain_returns_400(client):
    page = client.get("/").data.decode()
    code = page.split('<option value="')[2].split('"')[0]
    r = client.get(f"/api/recommend?event={code}&captain=999999")
    assert r.status_code == 400
