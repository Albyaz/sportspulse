import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console
import time

load_dotenv()

console = Console()

FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

# --- TOP 12 LEAGUES CONFIG ---
# Season 2024 = the 2024/25 season for European leagues
# MLS and Brazilian Serie A run calendar year — also 2024
# Saudi Pro League runs Aug-May — also 2024
# World Cup 2026 is a standalone tournament — season 2026
LEAGUES = [
    {"id": 39,  "name": "Premier League",    "season": 2024},
    {"id": 140, "name": "La Liga",           "season": 2024},
    {"id": 78,  "name": "Bundesliga",        "season": 2024},
    {"id": 135, "name": "Serie A",           "season": 2024},
    {"id": 61,  "name": "Ligue 1",           "season": 2024},
    {"id": 2,   "name": "Champions League",  "season": 2024},
    {"id": 88,  "name": "Eredivisie",        "season": 2024},
    {"id": 94,  "name": "Primeira Liga",     "season": 2024},
    {"id": 71,  "name": "Brazilian Serie A", "season": 2024},
    {"id": 307, "name": "Saudi Pro League",  "season": 2024},
    {"id": 253, "name": "MLS",               "season": 2024},
    {"id": 1,   "name": "World Cup 2026",    "season": 2026},
]


def get_engine():
    return create_engine(DATABASE_URL)


def create_tables(engine):
    """Create Bronze layer tables if they don't exist."""
    console.print("[bold yellow]Creating Bronze layer tables...[/bold yellow]")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bronze_fixtures (
                id SERIAL PRIMARY KEY,
                fixture_id INTEGER,
                referee TEXT,
                timezone TEXT,
                date TIMESTAMP,
                venue_name TEXT,
                venue_city TEXT,
                status_long TEXT,
                status_short TEXT,
                status_elapsed INTEGER,
                league_id INTEGER,
                league_name TEXT,
                league_country TEXT,
                league_season INTEGER,
                league_round TEXT,
                home_team_id INTEGER,
                home_team_name TEXT,
                home_team_logo TEXT,
                away_team_id INTEGER,
                away_team_name TEXT,
                away_team_logo TEXT,
                home_goals INTEGER,
                away_goals INTEGER,
                home_halftime INTEGER,
                away_halftime INTEGER,
                ingested_at TIMESTAMP DEFAULT NOW()
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bronze_teams (
                id SERIAL PRIMARY KEY,
                team_id INTEGER,
                team_name TEXT,
                team_code TEXT,
                team_country TEXT,
                team_founded INTEGER,
                team_logo TEXT,
                league_id INTEGER,
                league_name TEXT,
                season INTEGER,
                ingested_at TIMESTAMP DEFAULT NOW()
            );
        """))

        conn.commit()

    console.print("[bold green]✓ Tables ready[/bold green]")


def fetch_fixtures(league):
    """Fetch all fixtures for a league."""
    response = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"league": league["id"], "season": league["season"]}
    )

    if response.status_code != 200:
        console.print(f"[red]  ✗ HTTP {response.status_code}[/red]")
        return []

    data = response.json()
    errors = data.get("errors", {})
    if errors:
        console.print(f"[red]  ✗ Error: {errors}[/red]")
        return []

    count = data.get("results", 0)
    console.print(f"[green]  ✓ {count} fixtures[/green]")
    return data.get("response", [])


def fetch_teams(league):
    """Fetch all teams for a league."""
    response = requests.get(
        f"{BASE_URL}/teams",
        headers=HEADERS,
        params={"league": league["id"], "season": league["season"]}
    )

    if response.status_code != 200:
        console.print(f"[red]  ✗ HTTP {response.status_code}[/red]")
        return []

    data = response.json()
    errors = data.get("errors", {})
    if errors:
        console.print(f"[red]  ✗ Error: {errors}[/red]")
        return []

    count = data.get("results", 0)
    console.print(f"[green]  ✓ {count} teams[/green]")
    return data.get("response", [])


def transform_fixtures(raw_fixtures):
    """Transform raw fixture data into flat DataFrame."""
    records = []

    for item in raw_fixtures:
        fixture = item.get("fixture", {})
        lg = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        score = item.get("score", {})

        records.append({
            "fixture_id": fixture.get("id"),
            "referee": fixture.get("referee"),
            "timezone": fixture.get("timezone"),
            "date": fixture.get("date"),
            "venue_name": fixture.get("venue", {}).get("name"),
            "venue_city": fixture.get("venue", {}).get("city"),
            "status_long": fixture.get("status", {}).get("long"),
            "status_short": fixture.get("status", {}).get("short"),
            "status_elapsed": fixture.get("status", {}).get("elapsed"),
            "league_id": lg.get("id"),
            "league_name": lg.get("name"),
            "league_country": lg.get("country"),
            "league_season": lg.get("season"),
            "league_round": lg.get("round"),
            "home_team_id": teams.get("home", {}).get("id"),
            "home_team_name": teams.get("home", {}).get("name"),
            "home_team_logo": teams.get("home", {}).get("logo"),
            "away_team_id": teams.get("away", {}).get("id"),
            "away_team_name": teams.get("away", {}).get("name"),
            "away_team_logo": teams.get("away", {}).get("logo"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "home_halftime": score.get("halftime", {}).get("home"),
            "away_halftime": score.get("halftime", {}).get("away"),
        })

    return pd.DataFrame(records)


def transform_teams(raw_teams, league):
    """Transform raw team data into flat DataFrame."""
    records = []

    for item in raw_teams:
        team = item.get("team", {})
        records.append({
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "team_code": team.get("code"),
            "team_country": team.get("country"),
            "team_founded": team.get("founded"),
            "team_logo": team.get("logo"),
            "league_id": league["id"],
            "league_name": league["name"],
            "season": league["season"],
        })

    return pd.DataFrame(records)


def clear_league_data(engine, league):
    """Clear existing data for this league/season before reloading."""
    with engine.connect() as conn:
        conn.execute(text(
            "DELETE FROM bronze_fixtures WHERE league_id = :id AND league_season = :season"
        ), {"id": league["id"], "season": league["season"]})
        conn.execute(text(
            "DELETE FROM bronze_teams WHERE league_id = :id AND season = :season"
        ), {"id": league["id"], "season": league["season"]})
        conn.commit()


def load_to_bronze(df, table_name, engine):
    """Append DataFrame to Bronze table."""
    if df.empty:
        console.print(f"[yellow]  ⚠ No data[/yellow]")
        return
    df.to_sql(table_name, engine, if_exists="append", index=False)


def run_pipeline():
    """Run the full multi-league ingestion pipeline."""
    console.print("\n[bold magenta]========================================[/bold magenta]")
    console.print("[bold magenta]  SportsPulse — Multi-League Ingestion  [/bold magenta]")
    console.print("[bold magenta]========================================[/bold magenta]\n")

    start_time = datetime.now()
    engine = get_engine()
    create_tables(engine)

    total_fixtures = 0
    total_teams = 0
    requests_used = 0

    for league in LEAGUES:
        console.print(f"\n[bold yellow]▶ {league['name']} (ID:{league['id']}, Season:{league['season']})[/bold yellow]")
        clear_league_data(engine, league)

        # Fixtures
        console.print(f"[cyan]  Fetching fixtures...[/cyan]", end=" ")
        raw_fixtures = fetch_fixtures(league)
        requests_used += 1
        time.sleep(0.5)

        if raw_fixtures:
            fixtures_df = transform_fixtures(raw_fixtures)
            load_to_bronze(fixtures_df, "bronze_fixtures", engine)
            total_fixtures += len(fixtures_df)
            console.print(f"[green]  ✓ Loaded {len(fixtures_df)} fixtures[/green]")

        # Teams
        console.print(f"[cyan]  Fetching teams...[/cyan]", end=" ")
        raw_teams = fetch_teams(league)
        requests_used += 1
        time.sleep(0.5)

        if raw_teams:
            teams_df = transform_teams(raw_teams, league)
            load_to_bronze(teams_df, "bronze_teams", engine)
            total_teams += len(teams_df)
            console.print(f"[green]  ✓ Loaded {len(teams_df)} teams[/green]")

    elapsed = (datetime.now() - start_time).seconds

    console.print(f"\n[bold green]========================================[/bold green]")
    console.print(f"[bold green]✓ All leagues ingested in {elapsed}s[/bold green]")
    console.print(f"[bold green]✓ Total fixtures loaded: {total_fixtures}[/bold green]")
    console.print(f"[bold green]✓ Total teams loaded:    {total_teams}[/bold green]")
    console.print(f"[bold green]✓ API requests used:     {requests_used}/100[/bold green]")
    console.print(f"[bold green]✓ Bronze layer ready in Neon PostgreSQL[/bold green]")
    console.print(f"[bold cyan]  World Cup 2026 data populates from June 11[/bold cyan]\n")


if __name__ == "__main__":
    run_pipeline()