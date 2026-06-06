import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
from rich.console import Console

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import pickle

load_dotenv()

console = Console()
DATABASE_URL = os.getenv("DATABASE_URL")

FEATURE_COLS = [
    "home_win_rate", "home_draw_rate", "home_loss_rate",
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_goal_diff", "home_home_win_rate", "home_over_2_5_rate",
    "home_form_score", "home_points",
    "away_win_rate", "away_draw_rate", "away_loss_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_goal_diff", "away_away_win_rate", "away_over_2_5_rate",
    "away_form_score", "away_points",
    "win_rate_diff", "goals_scored_diff", "goals_conceded_diff",
    "goal_diff_diff", "form_diff", "points_diff",
    "h2h_total_matches", "h2h_home_win_rate", "h2h_away_win_rate",
    "h2h_draw_rate", "h2h_avg_goals",
    "kick_off_hour", "day_of_week",
]


def get_engine():
    return create_engine(DATABASE_URL)


def load_features(engine):
    """Load feature matrix from database."""
    console.print("[cyan]Loading feature matrix...[/cyan]")
    df = pd.read_sql("SELECT * FROM ml_features", engine)
    console.print(f"[green]✓ Loaded {len(df)} rows[/green]")
    return df


def prepare_data(df):
    """Prepare X and y for training."""
    console.print("[cyan]Preparing training data...[/cyan]")

    df = df.dropna(subset=FEATURE_COLS + ["result"])

    X = df[FEATURE_COLS].fillna(0)
    y_raw = df["result"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    console.print(f"[green]✓ X shape: {X.shape}[/green]")
    console.print(f"[green]✓ Classes: {list(le.classes_)}[/green]")

    return X, y, le, df


def train_model(X, y):
    """Train XGBoost classifier."""
    console.print("\n[cyan]Splitting data 80/20...[/cyan]")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    console.print(f"  Train: {len(X_train)} samples")
    console.print(f"  Test:  {len(X_test)} samples")

    console.print("\n[cyan]Training XGBoost model...[/cyan]")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)
    console.print("[green]✓ Model trained[/green]")

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, le):
    """Evaluate model performance."""
    console.print("\n[bold cyan]--- Model Evaluation ---[/bold cyan]")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    console.print(f"\n[bold green]Overall Accuracy: {accuracy*100:.2f}%[/bold green]")

    console.print("\n[bold]Classification Report:[/bold]")
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=3
    )
    console.print(report)

    console.print("[bold]Confusion Matrix:[/bold]")
    cm = confusion_matrix(y_test, y_pred)
    classes = le.classes_
    header = f"{'':>12}" + "".join(f"{c:>12}" for c in classes)
    console.print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = f"  {classes[i]:>12}" + "".join(f"{v:>12}" for v in row)
        console.print(row_str)

    return accuracy, y_pred


def get_feature_importance(model, top_n=10):
    """Display top feature importances."""
    console.print(f"\n[bold cyan]Top {top_n} Feature Importances[/bold cyan]")

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)

    for feat, imp in feat_imp.head(top_n).items():
        bar = "█" * int(imp * 100)
        console.print(f"  {feat:<35} {imp:.4f} {bar}")

    return feat_imp


def save_predictions(model, df, le, engine):
    """Generate and save predictions for all fixtures."""
    console.print("\n[cyan]Generating predictions for all fixtures...[/cyan]")

    X_all = df[FEATURE_COLS].fillna(0)
    proba = model.predict_proba(X_all)
    pred_labels = le.inverse_transform(model.predict(X_all))

    classes = le.classes_
    proba_df = pd.DataFrame(proba, columns=[f"prob_{c.lower()}" for c in classes])

    results_df = pd.DataFrame({
        "fixture_id": df["fixture_id"].values,
        "league_name": df["league_name"].values,
        "home_team": df["home_team_name"].values,
        "away_team": df["away_team_name"].values,
        "actual_result": df["result"].values,
        "predicted_result": pred_labels,
        "correct": (df["result"].values == pred_labels).astype(int),
    })

    results_df = pd.concat([results_df, proba_df], axis=1)

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ml_predictions"))
        conn.commit()

    results_df.to_sql("ml_predictions", engine, if_exists="replace", index=False)
    console.print(f"[green]✓ Saved {len(results_df)} predictions to ml_predictions table[/green]")

    return results_df


def save_model(model, le):
    """Save model and label encoder to disk."""
    os.makedirs("ml/models", exist_ok=True)
    with open("ml/models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("ml/models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    console.print("[green]✓ Model saved to ml/models/xgboost_model.pkl[/green]")


def run_predictor():
    """Run the full ML training and prediction pipeline."""
    console.print("\n[bold magenta]========================================[/bold magenta]")
    console.print("[bold magenta]  SportsPulse — XGBoost Outcome Predictor[/bold magenta]")
    console.print("[bold magenta]========================================[/bold magenta]\n")

    start_time = datetime.now()
    engine = get_engine()

    # Load and prepare
    df = load_features(engine)
    X, y, le, df_clean = prepare_data(df)

    # Train
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # Evaluate
    accuracy, y_pred = evaluate_model(model, X_test, y_test, le)

    # Feature importance
    get_feature_importance(model)

    # Save predictions
    save_predictions(model, df_clean, le, engine)

    # Save model
    save_model(model, le)

    elapsed = (datetime.now() - start_time).seconds
    console.print(f"\n[bold green]✓ ML pipeline complete in {elapsed}s[/bold green]")
    console.print(f"[bold green]✓ Accuracy: {accuracy*100:.2f}%[/bold green]")
    console.print(f"[bold green]✓ Predictions saved to Neon PostgreSQL[/bold green]")
    console.print(f"[bold green]✓ Model saved to ml/models/xgboost_model.pkl[/bold green]\n")


if __name__ == "__main__":
    run_predictor()