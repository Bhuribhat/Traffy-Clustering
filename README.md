# Traffy Clustering Project

__Created by__  

- 6330440921 Bhuribhat Ratanasanguanvongs
- 6330301321 Panithi Khamwangyang
- 6330305921 Pras Pitasawad


## Initializing Environment

If you are using linux os, please do the following:

```sh
>> mkdir ./dags ./logs ./plugins
>> echo -e  "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env
```

For other operating systems, you may get a warning that __AIRFLOW_UID__ is not set, but you can safely ignore it. You can also manually create an __.env__ file in the same folder as __docker-compose.yaml__ with this content to get rid of the warning:  

```sh
AIRFLOW_UID=50000
```

- __./dags__    - you can put your DAG files here.
- __./logs__    - contains logs from task execution and scheduler.
- __./plugins__ - you can put your [custom plugins](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/plugins.html) here.


## Run Airflow

```sh
>> docker-compose up airflow-init   # run database and create first user account
>> docker-compose up -d             # run container in background
```


Run `docker ps` to check the condition of the containers and make sure that no containers are in unhealthy condition:

```sh
>> docker ps
CONTAINER ID   IMAGE                  COMMAND                  CREATED          STATUS                    PORTS                              NAMES
247ebe6cf87a   apache/airflow:2.6.0   "/usr/bin/dumb-init …"   3 minutes ago    Up 3 minutes (healthy)    8080/tcp                           compose_airflow-worker_1
ed9b09fc84b1   apache/airflow:2.6.0   "/usr/bin/dumb-init …"   3 minutes ago    Up 3 minutes (healthy)    8080/tcp                           compose_airflow-scheduler_1
7cb1fb603a98   apache/airflow:2.6.0   "/usr/bin/dumb-init …"   3 minutes ago    Up 3 minutes (healthy)    0.0.0.0:8080->8080/tcp             compose_airflow-webserver_1
74f3bbe506eb   postgres:13            "docker-entrypoint.s…"   18 minutes ago   Up 17 minutes (healthy)   5432/tcp                           compose_postgres_1
0bd6576d23cb   redis:latest           "docker-entrypoint.s…"   10 hours ago     Up 17 minutes (healthy)   0.0.0.0:6379->6379/tcp   
```


## Open Airflow UI

The webserver is available at: `http://localhost:8080`  
The default account has the login __airflow__ and the password __airflow__. 


## Close Airflow Docker

```sh
>> docker-compose down -v
```
