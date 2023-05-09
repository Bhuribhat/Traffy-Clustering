from airflow import DAG
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'pras',
    # 'depends_on_past': False,
    # 'start_date': datetime(2023, 5, 9),
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
  # 1. Check if the API is up
  task_is_api_active = HttpSensor(
      task_id='is_api_active',
      http_conn_id='',
      endpoint='https://publicapi.traffy.in.th/ud/search'
  )


  task_get_json_data_from_fondue = SimpleHttpOperator(
      task_id='get_json_data_from_fondue',
      http_conn_id='',  # specify the connection ID for your API
      endpoint='https://publicapi.traffy.in.th/ud/search',  # specify the API endpoint to retrieve JSON data
      method='GET',
      headers={"Content-Type": "application/json"},
      dag=dag,
  )
  task_is_api_active >> task_get_json_data_from_fondue

# dag = DAG('get_json_data', default_args=default_args, schedule_interval=timedelta(days=1))