from airflow import DAG
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.email_operator import EmailOperator
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
    # Check if the API is up
    task_is_api_active = HttpSensor(
        task_id='is_api_active',
        http_conn_id='',
        endpoint='https://publicapi.traffy.in.th/ud/search'
    )

    task_get_json_data_from_fondue = SimpleHttpOperator(
        task_id='get_json_data_from_fondue',
        http_conn_id='',  										# specify the connection ID for your API
        endpoint='https://publicapi.traffy.in.th/ud/search',  	# specify the API endpoint to retrieve JSON data
        method='GET',
        headers={"Content-Type": "application/json"},
        dag=dag,
    )

    task_send_email = EmailOperator(
        task_id='send_email',
        to=['Bhuribhat@gmail.com'],
        subject='Your traffy data is ready',
        html_content='Please check your dashboard.'
    )

    task_is_api_active >> task_get_json_data_from_fondue >> task_send_email

# dag = DAG('get_json_data', default_args=default_args, schedule_interval=timedelta(days=1))
