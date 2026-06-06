import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_engine():
    return create_engine(DATABASE_URL)


def create_silver_tables(engine):
    """Create Silver layer tables."""
    console.print("[bold yellow]Creating Silver layer tables...[/bold yellow]")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS silver_fixtures (
                id SERIAL PRIMARY KEY,
                fixture_id INTEGER UNIQUE,
                date TIMESTAMP,
                day_of_week TEXT,
                kick_off_hour INTEGER,
                venue_name TEXT,
                venue_city TEXT,
                league_id INTEGER,
                league_name TEXT,
                league_season INTEGER,
                league_round TEXT,
                round_number INTEGER,
                home_team_id INTEGER,
                home_team_name TEXT,
                away_team_id INTEGER,
                away_team_name TEXT,
                home_goals INTEGER,
                away_goals INTEGER,
                total_goals INTEGER,
                goal_difference INTEGER,
                result TEXT,
                home_win INTEGER,
                away_win INTEGER,
                draw INTEGER,
                both_teams_scored INTEGER,
                over_2_5 INTEGER,
                status TEXT,
                is_finished INTEGER,
                ingested_at TIMESTAMP DEFAULT NOW()
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS silver_teams (
                id SERIAL PRIMARY KEY,
                team_id INTEGER UNIQUE,
                team_name TEXT,
                team_code TEXT,
                team_country TEXT,
                team_founded INTEGER,
                team_age INTEGER,
                league_id INTEGER,
                league_name TEXT,
                season INTEGER,
                ingested_at TIMESTAMP DEFAULT NOW()
            );
        """))

        conn.commit()

    console.print("[bold green]✓ Silver tables ready[/bold green]")


def extract_bronze_fixtures(engine):
    """Pull raw fixtures from Bronze layer."""
    console.print("[cyan]Extracting from bronze_fixtures...[/cyan]")
    df = pd.read_sql("SELECT * FROM bronze_fixtures", engine)
    console.print(f"[green]✓ Extracted {len(df)} records[/green]")
    return df


def extract_bronze_teams(engine):
    """Pull raw teams from Bronze layer."""
    console.print("[cyan]Extracting from bronze_teams...[/cyan]")
    df = pd.read_sql("SELECT * FROM bronze_teams", engine)
    console.print(f"[green]✓ Extracted {len(df)} records[/green]")
    return df


def transform_fixtures(df):
    """Clean and enrich fixtures data."""
    console.print("[cyan]Transforming fixtures...[/cyan]")

    # Drop duplicates
    df = df.drop_duplicates(subset=["fixture_id"])

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    # Extract time features
    df["day_of_week"] = df["date"].dt.day_name()
    df["kick_off_hour"] = df["date"].dt.hour

    # Extract round number from round string e.g. "Regular Season - 1"
    df["round_number"] = df["league_round"].str.extract(r"(\d+)").astype(float)

    # Only process finished matches for goals
    finished_mask = df["status_short"] == "FT"

    # Fill goals with 0 for finished matches, keep NaN for unplayed
    df.loc[finished_mask, "home_goals"] = df.loc[finished_mask, "home_goals"].fillna(0)
    df.loc[finished_mask, "away_goals"] = df.loc[finished_mask, "away_goals"].fillna(0)

    # Goal metrics (only for finished matches)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_difference"] = df["home_goals"] - df["away_goals"]

    # Match result
    def get_result(row):
        if pd.isna(row["home_goals"]) or pd.isna(row["away_goals"]):
            return "TBD"
        if row["home_goals"] > row["away_goals"]:
            return "HOME_WIN"
        elif row["away_goals"] > row["home_goals"]:
            return "AWAY_WIN"
        else:
            return "DRAW"

    df["result"] = df.apply(get_result, axis=1)

    # Binary outcome columns (for ML later)
    df["home_win"] = (df["result"] == "HOME_WIN").astype(int)
    df["away_win"] = (df["result"] == "AWAY_WIN").astype(int)
    df["draw"] = (df["result"] == "DRAW").astype(int)

    # Both teams scored
    df["both_teams_scored"] = (
        (df["home_goals"] > 0) & (df["away_goals"] > 0)
    ).astype(int)

    # Over 2.5 goals
    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)

    # Is finished flag
    df["is_finished"] = (df["status_short"] == "FT").astype(int)
    df["status"] = df["status_long"]

    # Select final columns
    silver_cols = [
        "fixture_id", "date", "day_of_week", "kick_off_hour",
        "venue_name", "venue_city", "league_id", "league_name",
        "league_season", "league_round", "round_number",
        "home_team_id", "home_team_name", "away_team_id", "away_team_name",
        "home_goals", "away_goals", "total_goals", "goal_difference",
        "result", "home_win", "away_win", "draw",
        "both_teams_scored", "over_2_5", "status", "is_finished"
    ]

    df = df[silver_cols]

    console.print(f"[green]✓ Transformed {len(df)} fixture records[/green]")
    return df


def transform_teams(df):
    """Clean and enrich teams data."""
    console.print("[cyan]Transforming teams...[/cyan]")

    df = df.drop_duplicates(subset=["team_id"])

    # Calculate team age
    current_year = datetime.now().year
    df["team_age"] = current_year - df["team_founded"]

    silver_cols = [
        "team_id", "team_name", "team_code", "team_country",
        "team_founded", "team_age", "league_id", "league_name", "season"
    ]

    df = df[silver_cols]

    console.print(f"[green]✓ Transformed {len(df)} team records[/green]")
    return df


def load_to_silver(df, table_name, engine, unique_col):
    """Load DataFrame into Silver layer, avoiding duplicates."""
    if df.empty:
        console.print(f"[yellow]⚠ No data to load into {table_name}[/yellow]")
        return

    # Get existing IDs
    existing = pd.read_sql(f"SELECT {unique_col} FROM {table_name}", engine)
    existing_ids = set(existing[unique_col].tolist())

    # Filter to new records only
    new_df = df[~df[unique_col].isin(existing_ids)]

    if new_df.empty:
        console.print(f"[yellow]⚠ All records already exist in {table_name}[/yellow]")
        return

    new_df.to_sql(table_name, engine, if_exists="append", index=False)
    console.print(f"[green]✓ Loaded {len(new_df)} new records into {table_name}[/green]")


def display_stats(df, title):
    """Display summary statistics."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")

    if "result" in df.columns:
        finished = df[df["is_finished"] == 1]
        if not finished.empty:
            console.print(f"  Total matches:     {len(df)}")
            console.print(f"  Finished:          {len(finished)}")
            console.print(f"  Home wins:         {finished['home_win'].sum()} ({finished['home_win'].mean()*100:.1f}%)")
            console.print(f"  Away wins:         {finished['away_win'].sum()} ({finished['away_win'].mean()*100:.1f}%)")
            console.print(f"  Draws:             {finished['draw'].sum()} ({finished['draw'].mean()*100:.1f}%)")
            console.print(f"  Avg goals/game:    {finished['total_goals'].mean():.2f}")
            console.print(f"  Over 2.5 goals:    {finished['over_2_5'].sum()} ({finished['over_2_5'].mean()*100:.1f}%)")
            console.print(f"  Both teams scored: {finished['both_teams_scored'].sum()} ({finished['both_teams_scored'].mean()*100:.1f}%)")

    if "team_age" in df.columns:
        console.print(f"  Total teams:       {len(df)}")
        console.print(f"  Oldest team:       {df.loc[df['team_age'].idxmax(), 'team_name']} ({df['team_age'].max()} years)")
        console.print(f"  Newest team:       {df.loc[df['team_age'].idxmin(), 'team_name']} ({df['team_age'].min()} years)")


def run_pipeline():
    """Run Bronze → Silver transformation pipeline."""
    console.print("\n[bold magenta]========================================[/bold magenta]")
    console.print("[bold magenta]  SportsPulse — Bronze → Silver Layer[/bold magenta]")
    console.print("[bold magenta]========================================[/bold magenta]\n")

    start_time = datetime.now()
    engine = get_engine()

    create_silver_tables(engine)

    # Fixtures
    console.print("\n[bold]--- Processing Fixtures ---[/bold]")
    bronze_fixtures = extract_bronze_fixtures(engine)
    silver_fixtures = transform_fixtures(bronze_fixtures)
    display_stats(silver_fixtures, "Fixtures Summary")
    load_to_silver(silver_fixtures, "silver_fixtures", engine, "fixture_id")

    # Teams
    console.print("\n[bold]--- Processing Teams ---[/bold]")
    bronze_teams = extract_bronze_teams(engine)
    silver_teams = transform_teams(bronze_teams)
    display_stats(silver_teams, "Teams Summary")
    load_to_silver(silver_teams, "silver_teams", engine, "team_id")

    elapsed = (datetime.now() - start_time).seconds
    console.print(f"\n[bold green]✓ Bronze → Silver complete in {elapsed}s[/bold green]")
    console.print(f"[bold green]✓ Clean data ready in Silver layer (Neon PostgreSQL)[/bold green]\n")


if __name__ == "__main__":
    run_pipeline()