"""
Inference layer for the FTC Alliance Forecaster.

This module is the bridge between the Colab notebook and the web app. The
notebook trains the model; this file only LOADS what the notebook exported and
reproduces the same feature math at prediction time.

Why it exists: in the notebook, recommend_partners() depends on globals such as
events_oh, SCORE_COMPONENTS and PRIMARY_COMPONENT, plus the in-memory
team_features table. None of those exist in a web process, so the notebook
exports a self-contained "bundle" and this module rebuilds the vectors from it.

Bundle contract (produced by colab/export_bundle.py):
    model_bundle/win_model.pkl         calibrated sklearn estimator
    model_bundle/team_features.parquet one row per (event_code, team)
    model_bundle/feature_columns.json  exact training column order
    model_bundle/metadata.json         season, region, metrics, versions
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent.parent / "model_bundle"


class BundleError(RuntimeError):
    """Raised when the exported bundle is missing or inconsistent."""


class Forecaster:
    """Loads one exported bundle and answers recommendation queries."""

    def __init__(self, bundle_dir: Path | str = BUNDLE_DIR):
        self.dir = Path(bundle_dir)
        self.model = None
        self.team_features = None
        self.feature_columns: list[str] = []
        self.metadata: dict = {}
        self._load()

    # ---------------------------------------------------------------- load
    def _load(self) -> None:
        needed = ["win_model.pkl", "team_features.parquet",
                  "feature_columns.json", "metadata.json"]
        missing = [f for f in needed if not (self.dir / f).exists()]
        if missing:
            raise BundleError(
                "Missing bundle file(s): " + ", ".join(missing)
                + f". Expected them in {self.dir}. Run colab/export_bundle.py "
                  "in your notebook and copy the model_bundle folder here."
            )

        with open(self.dir / "win_model.pkl", "rb") as fh:
            self.model = pickle.load(fh)

        self.team_features = pd.read_parquet(self.dir / "team_features.parquet")
        self.feature_columns = json.loads((self.dir / "feature_columns.json").read_text())
        self.metadata = json.loads((self.dir / "metadata.json").read_text())

        # The per-team columns the alliance vector is built from. Stored at
        # export time so the app never has to recompute numeric_feature_cols().
        self.base_columns = self.metadata.get("base_feature_columns", [])
        if not self.base_columns:
            raise BundleError("metadata.json has no 'base_feature_columns'. Re-export the bundle.")

        self.primary_component = self.metadata.get("primary_component", "totalPointsNp")
        self.opr_col = f"opr_{self.primary_component}"

        missing_cols = [c for c in self.base_columns if c not in self.team_features.columns]
        if missing_cols:
            raise BundleError(
                f"team_features.parquet is missing {len(missing_cols)} expected column(s), "
                f"first few: {missing_cols[:5]}"
            )

    # ------------------------------------------------------------- helpers
    def events(self) -> list[str]:
        """Event codes available for scouting, sorted."""
        return sorted(self.team_features["event_code"].dropna().unique().tolist())

    def teams_at(self, event_code: str) -> pd.DataFrame:
        """Teams at one event with rank and OPR, ordered by qualification rank."""
        ev = self.team_features[self.team_features["event_code"] == event_code]
        cols = [c for c in ("team", "qual_rank", self.opr_col) if c in ev.columns]
        return ev[cols].sort_values("qual_rank", na_position="last")

    def _lookup(self, event_code: str) -> pd.DataFrame:
        ev = self.team_features[self.team_features["event_code"] == event_code]
        if ev.empty:
            raise BundleError(f"No features for event {event_code}.")
        return ev.set_index("team")[self.base_columns].astype(float)

    def _alliance_vector(self, tf: pd.DataFrame, t1: int, t2: int) -> dict | None:
        """Same math as the notebook: sum_ and max_ over the two robots."""
        try:
            f1, f2 = tf.loc[int(t1)], tf.loc[int(t2)]
        except KeyError:
            return None
        if pd.isna(f1.get(self.opr_col)) and pd.isna(f2.get(self.opr_col)):
            return None
        vec: dict[str, float] = {}
        for c in self.base_columns:
            a, b = f1[c], f2[c]
            vec[f"sum_{c}"] = np.nansum([a, b])
            vec[f"max_{c}"] = np.nanmax([a, b]) if not (pd.isna(a) and pd.isna(b)) else np.nan
        return vec

    # ------------------------------------------------------------ main API
    def recommend(self, event_code: str, captain: int | None = None,
                  top_n: int = 10, available_only: bool = True) -> dict:
        """
        Rank candidate partners for a captain at one event.

        available_only mirrors a real draft: teams seeded ABOVE the captain
        would be captains themselves and cannot be picked.
        """
        ev = self.team_features[self.team_features["event_code"] == event_code].copy()
        if ev.empty:
            raise BundleError(f"No features for event {event_code}.")

        team_ids = set(ev["team"].dropna().astype(int))
        if captain is None:
            captain = int(ev.sort_values("qual_rank").iloc[0]["team"])
        captain = int(captain)
        if captain not in team_ids:
            raise BundleError(f"Team {captain} did not compete at {event_code}.")

        tf = self._lookup(event_code)
        pool = [t for t in team_ids if t != captain]

        if available_only and "qual_rank" in ev.columns:
            ranks = ev.set_index("team")["qual_rank"]
            cap_rank = ranks.get(captain)
            if pd.notna(cap_rank):
                pool = [t for t in pool
                        if pd.notna(ranks.get(t)) and float(ranks.get(t)) > float(cap_rank)]

        vectors = {t: v for t in pool if (v := self._alliance_vector(tf, captain, t)) is not None}
        if not vectors:
            raise BundleError(
                "No candidate alliances could be built. If 'available only' is on, "
                "this captain may be the lowest seed at the event."
            )

        vdf = pd.DataFrame(vectors).T
        reference = vdf.median(axis=0)  # a typical alliance at this event

        diff = (vdf.reindex(columns=self.feature_columns)
                - reference.reindex(self.feature_columns)).fillna(0.0)
        proba = self.model.predict_proba(diff)[:, 1]
        win_prob = dict(zip(vdf.index, proba))

        ann = ev.set_index("team")
        cap_opr = float(ann.loc[captain, self.opr_col]) if self.opr_col in ann else float("nan")

        rows = []
        for cand in vectors:
            c = ann.loc[cand]
            p_opr = float(c.get(self.opr_col, np.nan))
            rows.append({
                "candidate": int(cand),
                "partner_opr": round(p_opr, 1),
                "combined_opr": round(cap_opr + p_opr, 1),
                "win_prob_vs_field": round(float(win_prob[cand]), 3),
                "qual_rank": None if pd.isna(c.get("qual_rank")) else int(c.get("qual_rank")),
                "color_sorter": int(c.get("has_color_sorter", 0) or 0),
                "base": int(c.get("base_capable", 0) or 0),
                "strong_auto": int(c.get("strong_auto", 0) or 0),
                "clean": int(c.get("clean_robot", 0) or 0),
            })

        table = (pd.DataFrame(rows)
                 .sort_values("win_prob_vs_field", ascending=False)
                 .reset_index(drop=True))
        table.insert(0, "rank", table.index + 1)

        return {
            "event_code": event_code,
            "captain": captain,
            "captain_opr": None if pd.isna(cap_opr) else round(cap_opr, 1),
            "available_only": available_only,
            "n_candidates": int(len(table)),
            "recommendations": table.head(top_n).to_dict(orient="records"),
        }
