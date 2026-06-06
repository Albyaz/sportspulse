import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console

load_dotenv()

console = Console()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_engine():
    return create_engine(DATABASE_URL)


def create_gold_tables(engine):
    """Create Gold layer tables."""
    console.print("[bold yellow]Creating Gold layer tables...[/bold yellow]")

    with engine.connect() as conn:

        # Team performance aggregates
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold_team_stats (
                id SERIAL PRIMARY KEY,
                team_id INTEGER UNIQUE,
                team_name TEXT,
                league_id INTEGER,
                league_name TEXT,
                season INTEGER,
                matches_played INTEGER,
                wins INTEGER,
                draws INTEGER,
                losses INTEGER,
                win_rate FLOAT,
                draw_rate FLOAT,
                loss_rate FLOAT,
                goals_scored INTEGER,
                goals_conceded INTEGER,
                goal_difference INTEGER,
                avg_goals_scored FLOAT,
                avg_goals_conceded FLOAT,
                clean_sheets INTEGER,
                failed_to_score INTEGER,
                home_matches INTEGER,
                home_wins INTEGER,
                home_win_rate FLOAT,
                away_matches INTEGER,
                away_wins INTEGER,
                away_win_rate FLOAT,
                over_2_5_count INTEGER,
                over_2_5_rate FLOAT,
                both_teams_scored_count INTEGER,
                both_teams_scored_rate FLOAT,
                form_last_5 TEXT,
                points INTEGER,
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """))

        # League-wide trends by round
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold_league_trends (
                id SERIAL PRIMARY KEY,
                league_id INTEGER,
                league_name TEXT,
                season INTEGER,
                round_number INTEGER,
                matches_in_round INTEGER,
                avg_goals_per_game FLOAT,
                home_win_rate FLOAT,
                away_win_rate FLOAT,
                draw_rate FLOAT,
                over_2_5_rate FLOAT,
                both_teams_scored_rate FLOAT,
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """))

        # Head to head records
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold_head_to_head (
                id SERIAL PRIMARY KEY,
                team_a_id INTEGER,
                team_a_name TEXT,
                team_b_id INTEGER,
                team_b_name TEXT,
                total_matches INTEGER,
                team_a_wins INTEGER,
                team_b_wins INTEGER,
                draws INTEGER,
                team_a_win_rate FLOAT,
                team_b_win_rate FLOAT,
                avg_total_goals FLOAT,
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """))

        conn.commit()

    console.print("[bold green]✓ Gold tables ready[/bold green]")


def extract_silver_fixtures(engine):
    """Pull clean fixtures from Silver layer."""
    console.print("[cyan]Extracting from silver_fixtures...[/cyan]")
    df = pd.read_sql("SELECT * FROM silver_fixtures WHERE is_finished = 1", engine)
    console.print(f"[green]✓ Extracted {len(df)} finished fixtures[/green]")
    return df


def build_team_stats(df):
    """Build comprehensive team performance statistics."""
    console.print("[cyan]Building team stats...[/cyan]")

    teams = {}

    for _, row in df.iterrows():
        for side in ["home", "away"]:
            opponent = "away" if side == "home" else "home"
            team_id = row[f"{side}_team_id"]
            team_name = row[f"{side}_team_name"]

            if team_id not in teams:
                teams[team_id] = {
                    "team_id": team_id,
                    "team_name": team_name,
                    "league_id": row["league_id"],
                    "league_name": row["league_name"],
                    "season": row["league_season"],
                    "matches": [],
                    "home_matches": [],
                    "away_matches": [],
                }

            goals_for = row[f"{side}_goals"] or 0
            goals_against = row[f"{opponent}_goals"] or 0
            total = (goals_for or 0) + (goals_against or 0)

            if row["result"] == f"{side.upper()}_WIN":
                result = "W"
            elif row["result"] == "DRAW":
                result = "D"
            else:
                result = "L"

            match_data = {
                "result": result,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "clean_sheet": 1 if goals_against == 0 else 0,
                "failed_to_score": 1 if goals_for == 0 else 0,
                "over_2_5": 1 if total > 2.5 else 0,
                "both_scored": row["both_teams_scored"],
                "date": row["date"],
            }

            teams[team_id]["matches"].append(match_data)
            if side == "home":
                teams[team_id]["home_matches"].append(match_data)
            else:
                teams[team_id]["away_matches"].append(match_data)

    records = []
    for team_id, data in teams.items():
        matches = data["matches"]
        home = data["home_matches"]
        away = data["away_matches"]

        # Sort by date for form
        matches_sorted = sorted(matches, key=lambda x: x["date"])
        last_5 = [m["result"] for m in matches_sorted[-5:]]
        form = "".join(last_5)

        wins = sum(1 for m in matches if m["result"] == "W")
        draws = sum(1 for m in matches if m["result"] == "D")
        losses = sum(1 for m in matches if m["result"] == "L")
        played = len(matches)

        goals_scored = sum(m["goals_for"] for m in matches)
        goals_conceded = sum(m["goals_against"] for m in matches)

        home_wins = sum(1 for m in home if m["result"] == "W")
        away_wins = sum(1 for m in away if m["result"] == "W")

        over_2_5 = sum(m["over_2_5"] for m in matches)
        both_scored = sum(m["both_scored"] for m in matches)

        records.append({
            "team_id": team_id,
            "team_name": data["team_name"],
            "league_id": data["league_id"],
            "league_name": data["league_name"],
            "season": data["season"],
            "matches_played": played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": round(wins / played, 4) if played > 0 else 0,
            "draw_rate": round(draws / played, 4) if played > 0 else 0,
            "loss_rate": round(losses / played, 4) if played > 0 else 0,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goal_difference": goals_scored - goals_conceded,
            "avg_goals_scored": round(goals_scored / played, 2) if played > 0 else 0,
            "avg_goals_conceded": round(goals_conceded / played, 2) if played > 0 else 0,
            "clean_sheets": sum(m["clean_sheet"] for m in matches),
            "failed_to_score": sum(m["failed_to_score"] for m in matches),
            "home_matches": len(home),
            "home_wins": home_wins,
            "home_win_rate": round(home_wins / len(home), 4) if home else 0,
            "away_matches": len(away),
            "away_wins": away_wins,
            "away_win_rate": round(away_wins / len(away), 4) if away else 0,
            "over_2_5_count": over_2_5,
            "over_2_5_rate": round(over_2_5 / played, 4) if played > 0 else 0,
            "both_teams_scored_count": both_scored,
            "both_teams_scored_rate": round(both_scored / played, 4) if played > 0 else 0,
            "form_last_5": form,
            "points": (wins * 3) + draws,
        })

    df_out = pd.DataFrame(records).sort_values("points", ascending=False)
    console.print(f"[green]✓ Built stats for {len(df_out)} teams[/green]")
    return df_out


def build_league_trends(df):
    """Build league-wide trends aggregated by round."""
    console.print("[cyan]Building league trends by round...[/cyan]")

    df = df[df["round_number"].notna()].copy()
    df["round_number"] = df["round_number"].astype(int)

    grouped = df.groupby(["league_id", "league_name", "league_season", "round_number"])

    records = []
    for (league_id, league_name, season, round_num), group in grouped:
        records.append({
            "league_id": league_id,
            "league_name": league_name,
            "season": season,
            "round_number": round_num,
            "matches_in_round": len(group),
            "avg_goals_per_game": round(group["total_goals"].mean(), 2),
            "home_win_rate": round(group["home_win"].mean(), 4),
            "away_win_rate": round(group["away_win"].mean(), 4),
            "draw_rate": round(group["draw"].mean(), 4),
            "over_2_5_rate": round(group["over_2_5"].mean(), 4),
            "both_teams_scored_rate": round(group["both_teams_scored"].mean(), 4),
        })

    df_out = pd.DataFrame(records).sort_values("round_number")
    console.print(f"[green]✓ Built trends for {len(df_out)} rounds[/green]")
    return df_out


def build_head_to_head(df):
    """Build head to head records between all team pairs."""
    console.print("[cyan]Building head to head records...[/cyan]")

    h2h = {}

    for _, row in df.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]

        key = tuple(sorted([home_id, away_id]))
        team_a_id, team_b_id = key

        if key not in h2h:
            h2h[key] = {
                "team_a_id": team_a_id,
                "team_b_id": team_b_id,
                "team_a_name": row["home_team_name"] if home_id == team_a_id else row["away_team_name"],
                "team_b_name": row["away_team_name"] if away_id == team_b_id else row["home_team_name"],
                "matches": [],
            }

        if row["result"] == "HOME_WIN":
            winner = home_id
        elif row["result"] == "AWAY_WIN":
            winner = away_id
        else:
            winner = None

        h2h[key]["matches"].append({
            "winner": winner,
            "total_goals": row["total_goals"],
        })

    records = []
    for key, data in h2h.items():
        matches = data["matches"]
        team_a_id = data["team_a_id"]
        team_b_id = data["team_b_id"]
        total = len(matches)

        team_a_wins = sum(1 for m in matches if m["winner"] == team_a_id)
        team_b_wins = sum(1 for m in matches if m["winner"] == team_b_id)
        draws = sum(1 for m in matches if m["winner"] is None)
        avg_goals = round(sum(m["total_goals"] for m in matches) / total, 2) if total > 0 else 0

        records.append({
            "team_a_id": team_a_id,
            "team_a_name": data["team_a_name"],
            "team_b_id": team_b_id,
            "team_b_name": data["team_b_name"],
            "total_matches": total,
            "team_a_wins": team_a_wins,
            "team_b_wins": team_b_wins,
            "draws": draws,
            "team_a_win_rate": round(team_a_wins / total, 4) if total > 0 else 0,
            "team_b_win_rate": round(team_b_wins / total, 4) if total > 0 else 0,
            "avg_total_goals": avg_goals,
        })

    df_out = pd.DataFrame(records)
    console.print(f"[green]✓ Built {len(df_out)} head to head records[/green]")
    return df_out


def load_to_gold(df, table_name, engine):
    """Replace Gold table with fresh aggregated data."""
    if df.empty:
        console.print(f"[yellow]⚠ No data for {table_name}[/yellow]")
        return

    with engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY"))
        conn.commit()

    df.to_sql(table_name, engine, if_exists="append", index=False)
    console.print(f"[green]✓ Loaded {len(df)} records into {table_name}[/green]")


def display_top_teams(df, n=5):
    """Display top N teams by points."""
    console.print("\n[bold cyan]Top 5 Teams by Points[/bold cyan]")
    top = df.head(n)[["team_name", "matches_played", "wins", "draws", "losses", "goals_scored", "goals_conceded", "points", "form_last_5"]]
    for _, row in top.iterrows():
        console.print(
            f"  {row['team_name']:<22} "
            f"P:{row['matches_played']} "
            f"W:{row['wins']} D:{row['draws']} L:{row['losses']} "
            f"GD:{row['goals_scored']-row['goals_conceded']:+} "
            f"Pts:{row['points']} "
            f"Form:{row['form_last_5']}"
        )


def run_pipeline():
    """Run Silver → Gold transformation pipeline."""
    console.print("\n[bold magenta]========================================[/bold magenta]")
    console.print("[bold magenta]  SportsPulse — Silver → Gold Layer[/bold magenta]")
    console.print("[bold magenta]========================================[/bold magenta]\n")

    start_time = datetime.now()
    engine = get_engine()

    create_gold_tables(engine)

    console.print("\n[bold]--- Extracting Silver Data ---[/bold]")
    silver_df = extract_silver_fixtures(engine)

    console.print("\n[bold]--- Building Gold Aggregates ---[/bold]")
    team_stats_df = build_team_stats(silver_df)
    league_trends_df = build_league_trends(silver_df)
    h2h_df = build_head_to_head(silver_df)

    console.print("\n[bold]--- Loading to Gold Layer ---[/bold]")
    load_to_gold(team_stats_df, "gold_team_stats", engine)
    load_to_gold(league_trends_df, "gold_league_trends", engine)
    load_to_gold(h2h_df, "gold_head_to_head", engine)

    display_top_teams(team_stats_df)

    elapsed = (datetime.now() - start_time).seconds
    console.print(f"\n[bold green]✓ Silver → Gold complete in {elapsed}s[/bold green]")
    console.print(f"[bold green]✓ Analytics-ready data in Gold layer (Neon PostgreSQL)[/bold green]\n")


if __name__ == "__main__":
    run_pipeline()