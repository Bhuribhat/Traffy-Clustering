from airflow import DAG
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.email_operator import EmailOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from utils.load_data import load_data

default_args = {
    'owner': 'pras',
    # 'depends_on_past': False,
    # 'start_date': datetime(2023, 5, 9),
    'email_on_failure': False,
    'email': ['Bhuribhat@gmail.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='fondue_dag',
    default_args=default_args,
    description='Get data from fondue website',
    start_date=datetime(2023, 5, 9),
    schedule_interval='@daily'
) as dag:
    # Check if the API is up
    task_is_api_active = HttpSensor(
        task_id='is_api_active',
        http_conn_id='',
        endpoint='https://publicapi.traffy.in.th/share/teamchadchart/search'
    )

    task_get_json_data_from_fondue = PythonOperator(
        task_id='fetching_data',
        python_callable=load_data
    )

    task_is_api_active >> task_get_json_data_from_fondue

# dag = DAG('get_json_data', default_args=default_args, schedule_interval=timedelta(days=1))
