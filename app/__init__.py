"""Flask application factory for the FTC Alliance Forecaster."""
from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from .forecaster import BundleError, Forecaster


def create_app(bundle_dir: str | None = None) -> Flask:
    app = Flask(__name__)

    # Load the model once at startup, not per request. If the bundle is missing
    # the app still boots and shows a helpful message instead of crashing.
    try:
        app.forecaster = Forecaster(bundle_dir) if bundle_dir else Forecaster()
        app.load_error = None
    except BundleError as exc:
        app.forecaster = None
        app.load_error = str(exc)

    # ---------------------------------------------------------------- pages
    @app.route("/")
    def index():
        if app.forecaster is None:
            return render_template("error.html", message=app.load_error), 503
        return render_template(
            "index.html",
            events=app.forecaster.events(),
            meta=app.forecaster.metadata,
        )

    @app.route("/teams")
    def teams():
        """Populates the captain dropdown after an event is chosen."""
        if app.forecaster is None:
            return jsonify({"error": app.load_error}), 503
        event_code = request.args.get("event", "")
        try:
            df = app.forecaster.teams_at(event_code)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"teams": df.to_dict(orient="records")})

    @app.route("/recommend", methods=["POST"])
    def recommend():
        if app.forecaster is None:
            return render_template("error.html", message=app.load_error), 503

        event_code = (request.form.get("event") or "").strip()
        captain_raw = (request.form.get("captain") or "").strip()
        available_only = request.form.get("available_only") == "on"

        if not event_code:
            return render_template("error.html", message="Please choose an event."), 400
        try:
            captain = int(captain_raw) if captain_raw else None
        except ValueError:
            return render_template("error.html",
                                   message=f"'{captain_raw}' is not a valid team number."), 400

        try:
            result = app.forecaster.recommend(event_code, captain,
                                              top_n=10, available_only=available_only)
        except BundleError as exc:
            return render_template("error.html", message=str(exc)), 400

        return render_template("results.html", result=result, meta=app.forecaster.metadata)

    # ------------------------------------------------------------------ API
    @app.route("/api/recommend")
    def api_recommend():
        """JSON endpoint, e.g. /api/recommend?event=USNJUCLT2&captain=30682"""
        if app.forecaster is None:
            return jsonify({"error": app.load_error}), 503

        event_code = request.args.get("event", "")
        captain = request.args.get("captain", type=int)
        top_n = request.args.get("top_n", default=10, type=int)
        available_only = request.args.get("available_only", "true").lower() != "false"

        try:
            result = app.forecaster.recommend(event_code, captain, top_n, available_only)
        except BundleError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.route("/health")
    def health():
        ok = app.forecaster is not None
        return jsonify({
            "status": "ok" if ok else "no_model",
            "detail": app.load_error,
            "events_loaded": len(app.forecaster.events()) if ok else 0,
        }), (200 if ok else 503)

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=int(os.environ.get("PORT", 5000)))
