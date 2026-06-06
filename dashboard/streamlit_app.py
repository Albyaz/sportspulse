import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import requests

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
API_BASE = "http://localhost:5001"

st.set_page_config(
    page_title="SportsPulse",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a5298;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #90caf9;
        margin-top: 4px;
    }
    .predict-card {
        background: linear-gradient(135deg, #1a2744, #0d1b2a);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #2a5298;
        margin-top: 12px;
    }
    .win-prob {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4fc3f7;
        margin-bottom: 12px;
        border-bottom: 2px solid #2a5298;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=300)
def load_summary():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            fixtures = conn.execute(text("SELECT COUNT(*) FROM silver_fixtures")).scalar()
            teams = conn.execute(text("SELECT COUNT(*) FROM gold_team_stats")).scalar()
            leagues = conn.execute(text("SELECT COUNT(DISTINCT league_id) FROM silver_fixtures")).scalar()
            predictions = conn.execute(text("SELECT COUNT(*) FROM ml_predictions")).scalar()
            accuracy = conn.execute(text("SELECT ROUND(AVG(correct) * 100, 2) FROM ml_predictions")).scalar()
        return {
            "total_fixtures": int(fixtures),
            "total_teams": int(teams),
            "total_leagues": int(leagues),
            "total_predictions": int(predictions),
            "model_accuracy_pct": float(accuracy) if accuracy else 0,
        }
    except Exception as e:
        st.error(f"DB error: {e}")
        return {}


@st.cache_data(ttl=300)
def load_leagues():
    engine = get_engine()
    return pd.read_sql("""
        SELECT DISTINCT league_id, league_name
        FROM silver_fixtures
        ORDER BY league_name
    """, engine)


@st.cache_data(ttl=300)
def load_standings(league_id):
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT team_name, matches_played, wins, draws, losses,
               goals_scored, goals_conceded, goal_difference,
               points, form_last_5, win_rate, home_win_rate, away_win_rate,
               avg_goals_scored, avg_goals_conceded
        FROM gold_team_stats
        WHERE league_id = :lid
        ORDER BY points DESC, goal_difference DESC
    """), engine, params={"lid": league_id})


@st.cache_data(ttl=300)
def load_fixtures(league_id, limit=50):
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT fixture_id, date, home_team_name, away_team_name,
               home_goals, away_goals, result, status,
               league_name, league_round
        FROM silver_fixtures
        WHERE league_id = :lid AND is_finished = 1
        ORDER BY date DESC
        LIMIT :lim
    """), engine, params={"lid": league_id, "lim": limit})


@st.cache_data(ttl=300)
def load_trends(league_id):
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT * FROM gold_league_trends
        WHERE league_id = :lid
        ORDER BY round_number
    """), engine, params={"lid": league_id})


@st.cache_data(ttl=300)
def load_predictions(league_name, limit=50):
    engine = get_engine()
    return pd.read_sql(text("""
        SELECT * FROM ml_predictions
        WHERE league_name ILIKE :lg
        ORDER BY fixture_id DESC
        LIMIT :lim
    """), engine, params={"lg": f"%{league_name}%", "lim": limit})


@st.cache_data(ttl=300)
def load_all_teams():
    engine = get_engine()
    return pd.read_sql("""
        SELECT team_id, team_name, league_name
        FROM gold_team_stats
        ORDER BY team_name
    """, engine)


@st.cache_data(ttl=300)
def load_top_teams():
    engine = get_engine()
    return pd.read_sql("""
        SELECT team_name, league_name, points, wins, goal_difference, form_last_5
        FROM gold_team_stats
        ORDER BY points DESC
        LIMIT 15
    """, engine)


@st.cache_data(ttl=300)
def load_result_dist():
    engine = get_engine()
    return pd.read_sql("""
        SELECT league_name,
               SUM(home_win) as home_wins,
               SUM(away_win) as away_wins,
               SUM(draw) as draws
        FROM silver_fixtures
        WHERE is_finished = 1
        GROUP BY league_name
        ORDER BY league_name
    """, engine)


@st.cache_data(ttl=300)
def load_goals_data():
    engine = get_engine()
    return pd.read_sql("""
        SELECT league_name,
               ROUND(AVG(total_goals)::numeric, 2) as avg_goals,
               ROUND(AVG(CASE WHEN over_2_5 = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as over_2_5_pct,
               ROUND(AVG(CASE WHEN both_teams_scored = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as btts_pct
        FROM silver_fixtures
        WHERE is_finished = 1
        GROUP BY league_name
        ORDER BY avg_goals DESC
    """, engine)


def predict_match(home_id, away_id):
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={"home_team_id": int(home_id), "away_team_id": int(away_id)},
            timeout=10
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ SportsPulse")
    st.markdown("*Real-Time Football Analytics*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Standings", "📈 Trends", "🤖 Predictions", "🔮 Match Predictor"],
        label_visibility="collapsed"
    )

    st.divider()
    leagues_df = load_leagues()
    league_options = dict(zip(leagues_df["league_name"], leagues_df["league_id"]))
    selected_league_name = st.selectbox("Select League", list(league_options.keys()))
    selected_league_id = league_options[selected_league_name]

    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("- API-Football")
    st.markdown("- 12 Leagues")
    st.markdown("- XGBoost ML Model")


# ─────────────────────────────────────────
# OVERVIEW PAGE
# ─────────────────────────────────────────

if page == "🏠 Overview":
    st.markdown("# ⚽ SportsPulse Analytics")
    st.markdown("*Production-grade football data pipeline across 12 global leagues*")
    st.divider()

    summary = load_summary()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('total_fixtures', 0):,}</div>
            <div class="metric-label">Total Fixtures</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('total_teams', 0)}</div>
            <div class="metric-label">Teams</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('total_leagues', 0)}</div>
            <div class="metric-label">Leagues</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('total_predictions', 0):,}</div>
            <div class="metric-label">Predictions</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('model_accuracy_pct', 0):.1f}%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Top Teams Across All Leagues</div>', unsafe_allow_html=True)
        top_teams = load_top_teams()
        fig = px.bar(
            top_teams, x="points", y="team_name", color="league_name",
            orientation="h", title="Top 15 Teams by Points",
            labels={"points": "Points", "team_name": "Team"}, height=450,
        )
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Result Distribution by League</div>', unsafe_allow_html=True)
        result_dist = load_result_dist()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Home Win", x=result_dist["league_name"], y=result_dist["home_wins"], marker_color="#4fc3f7"))
        fig2.add_trace(go.Bar(name="Away Win", x=result_dist["league_name"], y=result_dist["away_wins"], marker_color="#f06292"))
        fig2.add_trace(go.Bar(name="Draw", x=result_dist["league_name"], y=result_dist["draws"], marker_color="#aed581"))
        fig2.update_layout(
            barmode="stack", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", title="Results by League", height=450, xaxis_tickangle=-45,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Goals Analysis Across Leagues</div>', unsafe_allow_html=True)
    goals_data = load_goals_data()
    fig3 = px.scatter(
        goals_data, x="avg_goals", y="over_2_5_pct", size="btts_pct",
        color="league_name", text="league_name",
        title="Avg Goals vs Over 2.5% (bubble size = BTTS%)",
        labels={"avg_goals": "Avg Goals/Game", "over_2_5_pct": "Over 2.5 Goals %"}, height=400,
    )
    fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
    fig3.update_traces(textposition="top center")
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────
# STANDINGS PAGE
# ─────────────────────────────────────────

elif page == "📊 Standings":
    st.markdown(f"# 📊 {selected_league_name} Standings")
    st.divider()

    standings = load_standings(selected_league_id)

    if standings.empty:
        st.warning("No standings data for this league.")
    else:
        standings.insert(0, "Pos", range(1, len(standings) + 1))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Teams", len(standings))
        with col2:
            st.metric("Avg Goals/Game", f"{standings['avg_goals_scored'].mean():.2f}")
        with col3:
            leader = standings.iloc[0]
            st.metric("League Leader", leader["team_name"], f"{leader['points']} pts")

        st.divider()

        display_cols = ["Pos", "team_name", "matches_played", "wins", "draws",
                        "losses", "goals_scored", "goals_conceded",
                        "goal_difference", "points", "form_last_5"]

        styled = standings[display_cols].rename(columns={
            "team_name": "Team", "matches_played": "P", "wins": "W",
            "draws": "D", "losses": "L", "goals_scored": "GF",
            "goals_conceded": "GA", "goal_difference": "GD",
            "points": "Pts", "form_last_5": "Form"
        })

        st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                standings.head(10), x="team_name",
                y=["home_win_rate", "away_win_rate"],
                title="Home vs Away Win Rate (Top 10)",
                labels={"value": "Win Rate", "team_name": "Team"},
                barmode="group", height=350,
            )
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                standings, x="goals_scored", y="goals_conceded",
                size="points", color="points", text="team_name",
                title="Goals Scored vs Conceded",
                color_continuous_scale="blues", height=350,
            )
            fig2.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white"
            )
            fig2.update_traces(textposition="top center", textfont_size=9)
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────
# TRENDS PAGE
# ─────────────────────────────────────────

elif page == "📈 Trends":
    st.markdown(f"# 📈 {selected_league_name} Trends")
    st.divider()

    trends = load_trends(selected_league_id)

    if trends.empty:
        st.warning("No trends data for this league.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                trends, x="round_number", y="avg_goals_per_game",
                title="Avg Goals Per Game by Round",
                labels={"round_number": "Round", "avg_goals_per_game": "Avg Goals"},
                markers=True, height=350,
            )
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
            fig.update_traces(line_color="#4fc3f7")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=trends["round_number"], y=trends["home_win_rate"] * 100, name="Home Win %", line=dict(color="#4fc3f7"), mode="lines+markers"))
            fig2.add_trace(go.Scatter(x=trends["round_number"], y=trends["away_win_rate"] * 100, name="Away Win %", line=dict(color="#f06292"), mode="lines+markers"))
            fig2.add_trace(go.Scatter(x=trends["round_number"], y=trends["draw_rate"] * 100, name="Draw %", line=dict(color="#aed581"), mode="lines+markers"))
            fig2.update_layout(
                title="Win/Draw Rates by Round", plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117", font_color="white",
                xaxis_title="Round", yaxis_title="Percentage %", height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig3 = px.area(
                trends, x="round_number", y="over_2_5_rate",
                title="Over 2.5 Goals Rate by Round",
                labels={"round_number": "Round", "over_2_5_rate": "Rate"}, height=350,
            )
            fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
            fig3.update_traces(fillcolor="rgba(79, 195, 247, 0.2)", line_color="#4fc3f7")
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.bar(
                trends, x="round_number", y="matches_in_round",
                title="Matches Per Round",
                labels={"round_number": "Round", "matches_in_round": "Matches"},
                height=350, color="matches_in_round", color_continuous_scale="blues",
            )
            fig4.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
            st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────
# PREDICTIONS PAGE
# ─────────────────────────────────────────

elif page == "🤖 Predictions":
    st.markdown(f"# 🤖 ML Predictions — {selected_league_name}")
    st.divider()

    preds = load_predictions(selected_league_name)

    if preds.empty:
        st.warning("No predictions for this league.")
    else:
        correct = preds["correct"].sum()
        total = len(preds)
        accuracy = correct / total * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Predictions", total)
        col2.metric("Correct", int(correct))
        col3.metric("Accuracy", f"{accuracy:.1f}%")

        st.divider()

        display = preds[[
            "home_team", "away_team", "actual_result",
            "predicted_result", "correct",
            "prob_home_win", "prob_away_win", "prob_draw"
        ]].copy()

        display["correct"] = display["correct"].map({1: "✅", 0: "❌"})
        display.columns = ["Home", "Away", "Actual", "Predicted", "✓", "P(Home)", "P(Away)", "P(Draw)"]

        st.dataframe(display, use_container_width=True, hide_index=True, height=500)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            result_counts = preds["predicted_result"].value_counts()
            fig = px.pie(
                values=result_counts.values, names=result_counts.index,
                title="Predicted Result Distribution",
                color_discrete_sequence=["#4fc3f7", "#f06292", "#aed581"], height=350,
            )
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "prob_home_win" in preds.columns:
                fig2 = px.histogram(
                    preds, x="prob_home_win", nbins=20,
                    title="Distribution of Home Win Probabilities",
                    labels={"prob_home_win": "Home Win Probability"},
                    color_discrete_sequence=["#4fc3f7"], height=350,
                )
                fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
                st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────
# MATCH PREDICTOR PAGE
# ─────────────────────────────────────────

elif page == "🔮 Match Predictor":
    st.markdown("# 🔮 Match Predictor")
    st.markdown("*Select any two teams and get an AI-powered outcome prediction*")
    st.divider()

    all_teams = load_all_teams()
    league_teams = all_teams[all_teams["league_name"] == selected_league_name]

    if league_teams.empty:
        st.warning("No teams found for this league.")
    else:
        team_options = dict(zip(league_teams["team_name"], league_teams["team_id"]))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🏠 Home Team")
            home_name = st.selectbox("Select Home Team", list(team_options.keys()), key="home")
        with col2:
            st.markdown("### ✈️ Away Team")
            away_options = [t for t in team_options.keys() if t != home_name]
            away_name = st.selectbox("Select Away Team", away_options, key="away")

        st.divider()

        if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
            home_id = team_options[home_name]
            away_id = team_options[away_name]

            with st.spinner("Running prediction..."):
                result = predict_match(home_id, away_id)

            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                probs = result.get("probabilities", {})
                home_prob = probs.get("HOME_WIN", 0)
                away_prob = probs.get("AWAY_WIN", 0)
                draw_prob = probs.get("DRAW", 0)
                predicted = result.get("predicted_result", "")

                col1, col2, col3 = st.columns(3)

                with col1:
                    color = "#4fc3f7" if predicted == "HOME_WIN" else "white"
                    st.markdown(f"""
                    <div class="predict-card" style="text-align:center;">
                        <div style="font-size:1rem;color:#90caf9;">🏠 {home_name}</div>
                        <div class="win-prob" style="color:{color};">{home_prob*100:.1f}%</div>
                        <div style="color:#aaa;font-size:0.85rem;">Win Probability</div>
                        {'<div style="color:#4fc3f7;font-weight:700;margin-top:8px;">✓ PREDICTED</div>' if predicted == "HOME_WIN" else ''}
                    </div>""", unsafe_allow_html=True)

                with col2:
                    color = "#aed581" if predicted == "DRAW" else "white"
                    st.markdown(f"""
                    <div class="predict-card" style="text-align:center;">
                        <div style="font-size:1rem;color:#90caf9;">🤝 Draw</div>
                        <div class="win-prob" style="color:{color};">{draw_prob*100:.1f}%</div>
                        <div style="color:#aaa;font-size:0.85rem;">Draw Probability</div>
                        {'<div style="color:#aed581;font-weight:700;margin-top:8px;">✓ PREDICTED</div>' if predicted == "DRAW" else ''}
                    </div>""", unsafe_allow_html=True)

                with col3:
                    color = "#f06292" if predicted == "AWAY_WIN" else "white"
                    st.markdown(f"""
                    <div class="predict-card" style="text-align:center;">
                        <div style="font-size:1rem;color:#90caf9;">✈️ {away_name}</div>
                        <div class="win-prob" style="color:{color};">{away_prob*100:.1f}%</div>
                        <div style="color:#aaa;font-size:0.85rem;">Win Probability</div>
                        {'<div style="color:#f06292;font-weight:700;margin-top:8px;">✓ PREDICTED</div>' if predicted == "AWAY_WIN" else ''}
                    </div>""", unsafe_allow_html=True)

                st.divider()

                fig = go.Figure(go.Bar(
                    x=[f"🏠 {home_name}", "🤝 Draw", f"✈️ {away_name}"],
                    y=[home_prob * 100, draw_prob * 100, away_prob * 100],
                    marker_color=["#4fc3f7", "#aed581", "#f06292"],
                    text=[f"{home_prob*100:.1f}%", f"{draw_prob*100:.1f}%", f"{away_prob*100:.1f}%"],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Outcome Probabilities",
                    yaxis_title="Probability %",
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font_color="white",
                    height=350,
                    yaxis=dict(range=[0, 110]),
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{home_name} Form:** `{result.get('home_form', 'N/A')}`")
                with col2:
                    st.markdown(f"**{away_name} Form:** `{result.get('away_form', 'N/A')}`")
                st.markdown(f"**H2H Matches on record:** {result.get('h2h_matches', 0)}")