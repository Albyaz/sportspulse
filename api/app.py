import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
import pickle

load_dotenv()

app = Flask(__name__)


class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

DATABASE_URL = os.getenv("DATABASE_URL")


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def load_model():
    with open("ml/models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ml/models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "SportsPulse API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.route("/leagues", methods=["GET"])
def get_leagues():
    engine = get_engine()
    df = pd.read_sql("""
        SELECT DISTINCT league_id, league_name, league_season, COUNT(*) as fixtures
        FROM silver_fixtures
        GROUP BY league_id, league_season, league_name
        ORDER BY fixtures DESC
    """, engine)
    return jsonify(df.to_dict(orient="records"))


@app.route("/fixtures", methods=["GET"])
def get_fixtures():
    league_id = request.args.get("league_id")
    status = request.args.get("status")
    limit = int(request.args.get("limit", 20))
    engine = get_engine()

    query = "SELECT * FROM silver_fixtures WHERE 1=1"
    params = {}

    if league_id:
        query += " AND league_id = :league_id"
        params["league_id"] = int(league_id)
    if status:
        query += " AND status ILIKE :status"
        params["status"] = f"%{status}%"

    query += " ORDER BY date DESC LIMIT :limit"
    params["limit"] = limit

    df = pd.read_sql(text(query), engine, params=params)
    df["date"] = df["date"].astype(str)
    return jsonify(df.to_dict(orient="records"))


@app.route("/teams", methods=["GET"])
def get_teams():
    league_id = request.args.get("league_id")
    engine = get_engine()

    query = "SELECT * FROM gold_team_stats WHERE 1=1"
    params = {}

    if league_id:
        query += " AND league_id = :league_id"
        params["league_id"] = int(league_id)

    query += " ORDER BY points DESC"
    df = pd.read_sql(text(query), engine, params=params)
    return jsonify(df.to_dict(orient="records"))


@app.route("/teams/<int:team_id>", methods=["GET"])
def get_team(team_id):
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM gold_team_stats WHERE team_id = :team_id"),
        engine, params={"team_id": team_id}
    )
    if df.empty:
        return jsonify({"error": "Team not found"}), 404
    return jsonify(df.to_dict(orient="records")[0])


@app.route("/standings", methods=["GET"])
def get_standings():
    league_id = request.args.get("league_id")
    if not league_id:
        return jsonify({"error": "league_id is required"}), 400

    engine = get_engine()
    df = pd.read_sql(
        text("""
            SELECT team_name, matches_played, wins, draws, losses,
                   goals_scored, goals_conceded, goal_difference,
                   points, form_last_5, win_rate, home_win_rate, away_win_rate
            FROM gold_team_stats
            WHERE league_id = :league_id
            ORDER BY points DESC, goal_difference DESC
        """),
        engine, params={"league_id": int(league_id)}
    )
    if df.empty:
        return jsonify({"error": "No data for this league"}), 404

    df["position"] = range(1, len(df) + 1)
    return jsonify(df.to_dict(orient="records"))


@app.route("/h2h", methods=["GET"])
def get_h2h():
    team_a = request.args.get("team_a")
    team_b = request.args.get("team_b")

    if not team_a or not team_b:
        return jsonify({"error": "team_a and team_b are required"}), 400

    engine = get_engine()
    df = pd.read_sql(
        text("""
            SELECT * FROM gold_head_to_head
            WHERE (team_a_id = :a AND team_b_id = :b)
               OR (team_a_id = :b AND team_b_id = :a)
        """),
        engine, params={"a": int(team_a), "b": int(team_b)}
    )
    if df.empty:
        return jsonify({"error": "No H2H record found"}), 404
    return jsonify(df.to_dict(orient="records")[0])


@app.route("/predictions", methods=["GET"])
def get_predictions():
    league_name = request.args.get("league")
    limit = int(request.args.get("limit", 20))
    engine = get_engine()

    query = "SELECT * FROM ml_predictions WHERE 1=1"
    params = {}

    if league_name:
        query += " AND league_name ILIKE :league"
        params["league"] = f"%{league_name}%"

    query += " ORDER BY fixture_id DESC LIMIT :limit"
    params["limit"] = limit

    df = pd.read_sql(text(query), engine, params=params)
    return jsonify(df.to_dict(orient="records"))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "home_team_id" not in data or "away_team_id" not in data:
        return jsonify({"error": "home_team_id and away_team_id required"}), 400

    home_id = data["home_team_id"]
    away_id = data["away_team_id"]
    engine = get_engine()

    home_df = pd.read_sql(
        text("SELECT * FROM gold_team_stats WHERE team_id = :id"),
        engine, params={"id": home_id}
    )
    away_df = pd.read_sql(
        text("SELECT * FROM gold_team_stats WHERE team_id = :id"),
        engine, params={"id": away_id}
    )
    h2h_df = pd.read_sql(
        text("""
            SELECT * FROM gold_head_to_head
            WHERE (team_a_id = :a AND team_b_id = :b)
               OR (team_a_id = :b AND team_b_id = :a)
        """),
        engine, params={"a": home_id, "b": away_id}
    )

    if home_df.empty or away_df.empty:
        return jsonify({"error": "One or both teams not found"}), 404

    home = home_df.iloc[0]
    away = away_df.iloc[0]

    def encode_form(form_str):
        if not form_str or not isinstance(form_str, str):
            return 0.0
        weights = {"W": 3, "D": 1, "L": 0}
        total = sum(weights.get(r, 0) for r in form_str)
        max_possible = len(form_str) * 3
        return round(total / max_possible, 4) if max_possible > 0 else 0.0

    home_form = encode_form(home.get("form_last_5", ""))
    away_form = encode_form(away.get("form_last_5", ""))

    if not h2h_df.empty:
        h2h = h2h_df.iloc[0]
        total = h2h["total_matches"]
        if h2h["team_a_id"] == home_id:
            h2h_home_wr = h2h["team_a_win_rate"]
            h2h_away_wr = h2h["team_b_win_rate"]
        else:
            h2h_home_wr = h2h["team_b_win_rate"]
            h2h_away_wr = h2h["team_a_win_rate"]
        h2h_draw_rate = round(h2h["draws"] / total, 4) if total > 0 else 0
        h2h_avg_goals = h2h["avg_total_goals"]
        h2h_total = total
    else:
        h2h_home_wr = h2h_away_wr = h2h_draw_rate = h2h_avg_goals = 0.0
        h2h_total = 0

    features = [[
        home["win_rate"], home["draw_rate"], home["loss_rate"],
        home["avg_goals_scored"], home["avg_goals_conceded"],
        home["goal_difference"], home["home_win_rate"], home["over_2_5_rate"],
        home_form, home["points"],
        away["win_rate"], away["draw_rate"], away["loss_rate"],
        away["avg_goals_scored"], away["avg_goals_conceded"],
        away["goal_difference"], away["away_win_rate"], away["over_2_5_rate"],
        away_form, away["points"],
        home["win_rate"] - away["win_rate"],
        home["avg_goals_scored"] - away["avg_goals_scored"],
        home["avg_goals_conceded"] - away["avg_goals_conceded"],
        home["goal_difference"] - away["goal_difference"],
        home_form - away_form,
        home["points"] - away["points"],
        h2h_total, h2h_home_wr, h2h_away_wr, h2h_draw_rate, h2h_avg_goals,
        12, 5,
    ]]

    model, le = load_model()
    proba = model.predict_proba(features)[0]
    pred = le.inverse_transform(model.predict(features))[0]
    classes = le.classes_

    return jsonify({
        "home_team": str(home["team_name"]),
        "away_team": str(away["team_name"]),
        "predicted_result": str(pred),
        "probabilities": {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, proba)
        },
        "home_form": str(home.get("form_last_5", "N/A")),
        "away_form": str(away.get("form_last_5", "N/A")),
        "h2h_matches": int(h2h_total),
    })


@app.route("/trends", methods=["GET"])
def get_trends():
    league_id = request.args.get("league_id")
    engine = get_engine()

    query = "SELECT * FROM gold_league_trends WHERE 1=1"
    params = {}

    if league_id:
        query += " AND league_id = :league_id"
        params["league_id"] = int(league_id)

    query += " ORDER BY league_id, round_number"
    df = pd.read_sql(text(query), engine, params=params)
    return jsonify(df.to_dict(orient="records"))


@app.route("/summary", methods=["GET"])
def get_summary():
    engine = get_engine()

    with engine.connect() as conn:
        fixtures = conn.execute(text("SELECT COUNT(*) FROM silver_fixtures")).scalar()
        teams = conn.execute(text("SELECT COUNT(*) FROM gold_team_stats")).scalar()
        leagues = conn.execute(text("SELECT COUNT(DISTINCT league_id) FROM silver_fixtures")).scalar()
        predictions = conn.execute(text("SELECT COUNT(*) FROM ml_predictions")).scalar()
        accuracy = conn.execute(text("""
            SELECT ROUND(AVG(correct) * 100, 2) FROM ml_predictions
        """)).scalar()

    return jsonify({
        "total_fixtures": int(fixtures),
        "total_teams": int(teams),
        "total_leagues": int(leagues),
        "total_predictions": int(predictions),
        "model_accuracy_pct": float(accuracy) if accuracy else 0,
        "last_updated": datetime.utcnow().isoformat(),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)