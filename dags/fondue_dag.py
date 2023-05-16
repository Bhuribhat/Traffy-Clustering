from airflow import DAG
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.email_operator import EmailOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from utils.load_data import load_data
from utils.clean_data import clean_data
from utils.clustering import kmean_cluster
from utils.visualize import visualize_data

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
    description='Visualize fondue data with poor people',
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

    task_clean_fondue_data = PythonOperator(
        task_id='clean_fondue_data',
        python_callable=clean_data
    )

    task_clusterring = PythonOperator(
        task_id='clustering',
        python_callable=kmean_cluster
    )

    task_visualzation = PythonOperator(
        task_id='visualize_data',
        python_callable=visualize_data
    )

    task_is_api_active >> task_get_json_data_from_fondue >> task_clean_fondue_data >> task_clusterring >> task_visualzation

# dag = DAG('get_json_data', default_args=default_args, schedule_interval=timedelta(days=1))
