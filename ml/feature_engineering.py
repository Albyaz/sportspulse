import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console

load_dotenv()

console = Console()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_engine():
    return create_engine(DATABASE_URL)


def load_gold_data(engine):
    """Load all Gold layer tables."""
    console.print("[cyan]Loading Gold layer data...[/cyan]")

    fixtures = pd.read_sql("SELECT * FROM silver_fixtures WHERE is_finished = 1", engine)
    team_stats = pd.read_sql("SELECT * FROM gold_team_stats", engine)
    h2h = pd.read_sql("SELECT * FROM gold_head_to_head", engine)

    console.print(f"[green]✓ Fixtures: {len(fixtures)}[/green]")
    console.print(f"[green]✓ Team stats: {len(team_stats)}[/green]")
    console.print(f"[green]✓ H2H records: {len(h2h)}[/green]")

    return fixtures, team_stats, h2h


def build_team_lookup(team_stats):
    """Build a quick lookup dict for team stats."""
    return team_stats.set_index("team_id").to_dict("index")


def get_team_features(team_id, lookup):
    """Extract features for a single team."""
    if team_id not in lookup:
        return {}

    t = lookup[team_id]
    return {
        "win_rate": t.get("win_rate", 0),
        "draw_rate": t.get("draw_rate", 0),
        "loss_rate": t.get("loss_rate", 0),
        "avg_goals_scored": t.get("avg_goals_scored", 0),
        "avg_goals_conceded": t.get("avg_goals_conceded", 0),
        "goal_difference": t.get("goal_difference", 0),
        "home_win_rate": t.get("home_win_rate", 0),
        "away_win_rate": t.get("away_win_rate", 0),
        "over_2_5_rate": t.get("over_2_5_rate", 0),
        "both_teams_scored_rate": t.get("both_teams_scored_rate", 0),
        "clean_sheets": t.get("clean_sheets", 0),
        "points": t.get("points", 0),
        "matches_played": t.get("matches_played", 0),
    }


def encode_form(form_str):
    """Convert form string like 'WWDLW' into numeric score."""
    if not form_str or not isinstance(form_str, str):
        return 0.0
    weights = {"W": 3, "D": 1, "L": 0}
    total = sum(weights.get(r, 0) for r in form_str)
    max_possible = len(form_str) * 3
    return round(total / max_possible, 4) if max_possible > 0 else 0.0


def get_h2h_features(home_id, away_id, h2h_df):
    """Get head to head stats between two teams."""
    mask = (
        ((h2h_df["team_a_id"] == home_id) & (h2h_df["team_b_id"] == away_id)) |
        ((h2h_df["team_a_id"] == away_id) & (h2h_df["team_b_id"] == home_id))
    )
    record = h2h_df[mask]

    if record.empty:
        return {
            "h2h_total_matches": 0,
            "h2h_home_win_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_avg_goals": 0.0,
        }

    row = record.iloc[0]
    total = row["total_matches"]

    # Align win rates to home/away perspective
    if row["team_a_id"] == home_id:
        home_wr = row["team_a_win_rate"]
        away_wr = row["team_b_win_rate"]
    else:
        home_wr = row["team_b_win_rate"]
        away_wr = row["team_a_win_rate"]

    draw_rate = round(row["draws"] / total, 4) if total > 0 else 0.0

    return {
        "h2h_total_matches": total,
        "h2h_home_win_rate": home_wr,
        "h2h_away_win_rate": away_wr,
        "h2h_draw_rate": draw_rate,
        "h2h_avg_goals": row["avg_total_goals"],
    }


def build_feature_matrix(fixtures, team_stats, h2h_df):
    """Build the full feature matrix for ML training."""
    console.print("[cyan]Building feature matrix...[/cyan]")

    lookup = build_team_lookup(team_stats)
    records = []

    for _, row in fixtures.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]

        home_feats = get_team_features(home_id, lookup)
        away_feats = get_team_features(away_id, lookup)
        h2h_feats = get_h2h_features(home_id, away_id, h2h_df)

        if not home_feats or not away_feats:
            continue

        # Get form scores
        home_form = encode_form(lookup.get(home_id, {}).get("form_last_5", ""))
        away_form = encode_form(lookup.get(away_id, {}).get("form_last_5", ""))

        record = {
            # Identifiers
            "fixture_id": row["fixture_id"],
            "league_id": row["league_id"],
            "league_name": row["league_name"],
            "home_team_id": home_id,
            "home_team_name": row["home_team_name"],
            "away_team_id": away_id,
            "away_team_name": row["away_team_name"],
            "date": row["date"],

            # Home team features
            "home_win_rate": home_feats["win_rate"],
            "home_draw_rate": home_feats["draw_rate"],
            "home_loss_rate": home_feats["loss_rate"],
            "home_avg_goals_scored": home_feats["avg_goals_scored"],
            "home_avg_goals_conceded": home_feats["avg_goals_conceded"],
            "home_goal_diff": home_feats["goal_difference"],
            "home_home_win_rate": home_feats["home_win_rate"],
            "home_over_2_5_rate": home_feats["over_2_5_rate"],
            "home_form_score": home_form,
            "home_points": home_feats["points"],

            # Away team features
            "away_win_rate": away_feats["win_rate"],
            "away_draw_rate": away_feats["draw_rate"],
            "away_loss_rate": away_feats["loss_rate"],
            "away_avg_goals_scored": away_feats["avg_goals_scored"],
            "away_avg_goals_conceded": away_feats["avg_goals_conceded"],
            "away_goal_diff": away_feats["goal_difference"],
            "away_away_win_rate": away_feats["away_win_rate"],
            "away_over_2_5_rate": away_feats["over_2_5_rate"],
            "away_form_score": away_form,
            "away_points": away_feats["points"],

            # Differential features (home minus away)
            "win_rate_diff": home_feats["win_rate"] - away_feats["win_rate"],
            "goals_scored_diff": home_feats["avg_goals_scored"] - away_feats["avg_goals_scored"],
            "goals_conceded_diff": home_feats["avg_goals_conceded"] - away_feats["avg_goals_conceded"],
            "goal_diff_diff": home_feats["goal_difference"] - away_feats["goal_difference"],
            "form_diff": home_form - away_form,
            "points_diff": home_feats["points"] - away_feats["points"],

            # H2H features
            "h2h_total_matches": h2h_feats["h2h_total_matches"],
            "h2h_home_win_rate": h2h_feats["h2h_home_win_rate"],
            "h2h_away_win_rate": h2h_feats["h2h_away_win_rate"],
            "h2h_draw_rate": h2h_feats["h2h_draw_rate"],
            "h2h_avg_goals": h2h_feats["h2h_avg_goals"],

            # Time features
            "kick_off_hour": row["kick_off_hour"],
            "day_of_week": row["date"].weekday() if pd.notna(row["date"]) else 0,

            # Target variable
            "result": row["result"],
        }

        records.append(record)

    df = pd.DataFrame(records)
    console.print(f"[green]✓ Feature matrix: {len(df)} rows x {len(df.columns)} columns[/green]")
    return df


def save_features(df, engine):
    """Save feature matrix to PostgreSQL."""
    console.print("[cyan]Saving feature matrix to database...[/cyan]")

    with engine.connect() as conn:
        from sqlalchemy import text
        conn.execute(text("DROP TABLE IF EXISTS ml_features"))
        conn.commit()

    df.to_sql("ml_features", engine, if_exists="replace", index=False)
    console.print(f"[green]✓ Saved {len(df)} rows to ml_features table[/green]")


def run_feature_engineering():
    """Run the full feature engineering pipeline."""
    console.print("\n[bold magenta]========================================[/bold magenta]")
    console.print("[bold magenta]  SportsPulse — Feature Engineering     [/bold magenta]")
    console.print("[bold magenta]========================================[/bold magenta]\n")

    start_time = datetime.now()
    engine = get_engine()

    # Load data
    fixtures, team_stats, h2h = load_gold_data(engine)

    # Build features
    feature_df = build_feature_matrix(fixtures, team_stats, h2h)

    # Show class distribution
    console.print("\n[bold cyan]Target Distribution (Result)[/bold cyan]")
    dist = feature_df["result"].value_counts()
    total = len(feature_df)
    for result, count in dist.items():
        console.print(f"  {result:<12} {count:>5} ({count/total*100:.1f}%)")

    # Show feature columns
    feature_cols = [c for c in feature_df.columns if c not in [
        "fixture_id", "league_id", "league_name",
        "home_team_id", "home_team_name",
        "away_team_id", "away_team_name",
        "date", "result"
    ]]
    console.print(f"\n[bold cyan]Feature Count: {len(feature_cols)} features[/bold cyan]")

    # Save
    save_features(feature_df, engine)

    elapsed = (datetime.now() - start_time).seconds
    console.print(f"\n[bold green]✓ Feature engineering complete in {elapsed}s[/bold green]")
    console.print(f"[bold green]✓ ml_features table ready in Neon PostgreSQL[/bold green]\n")


if __name__ == "__main__":
    run_feature_engineering()