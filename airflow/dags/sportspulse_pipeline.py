from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
import os

# Add project paths
sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'albright',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# ─────────────────────────────────────────
# DAG 1 — INGESTION
# ─────────────────────────────────────────

def run_ingestion():
    from ingestion.fixtures_extractor import run_pipeline
    run_pipeline()


with DAG(
    'sportspulse_ingestion',
    default_args=default_args,
    description='Ingest football data from API-Football',
    schedule_interval='0 6 * * *',  # Every day at 6am UTC
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=['sportspulse', 'ingestion'],
) as ingestion_dag:

    start = EmptyOperator(task_id='start')

    ingest_fixtures = PythonOperator(
        task_id='ingest_fixtures_and_teams',
        python_callable=run_ingestion,
    )

    end = EmptyOperator(task_id='end')

    start >> ingest_fixtures >> end


# ─────────────────────────────────────────
# DAG 2 — PROCESSING
# ─────────────────────────────────────────

def run_bronze_to_silver():
    from processing.bronze_to_silver import run_pipeline
    run_pipeline()


def run_silver_to_gold():
    from processing.silver_to_gold import run_pipeline
    run_pipeline()


with DAG(
    'sportspulse_processing',
    default_args=default_args,
    description='Process Bronze to Silver to Gold layers',
    schedule_interval='0 7 * * *',  # Every day at 7am UTC (after ingestion)
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=['sportspulse', 'processing'],
) as processing_dag:

    start = EmptyOperator(task_id='start')

    bronze_to_silver = PythonOperator(
        task_id='bronze_to_silver',
        python_callable=run_bronze_to_silver,
    )

    silver_to_gold = PythonOperator(
        task_id='silver_to_gold',
        python_callable=run_silver_to_gold,
    )

    end = EmptyOperator(task_id='end')

    start >> bronze_to_silver >> silver_to_gold >> end


# ─────────────────────────────────────────
# DAG 3 — ML
# ─────────────────────────────────────────

def run_feature_engineering():
    from ml.feature_engineering import run_feature_engineering
    run_feature_engineering()


def run_predictions():
    from ml.outcome_predictor import run_predictor
    run_predictor()


with DAG(
    'sportspulse_ml',
    default_args=default_args,
    description='Feature engineering and ML predictions',
    schedule_interval='0 8 * * *',  # Every day at 8am UTC (after processing)
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=['sportspulse', 'ml'],
) as ml_dag:

    start = EmptyOperator(task_id='start')

    feature_eng = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_feature_engineering,
    )

    predictions = PythonOperator(
        task_id='run_predictions',
        python_callable=run_predictions,
    )

    end = EmptyOperator(task_id='end')

    start >> feature_eng >> predictions >> end