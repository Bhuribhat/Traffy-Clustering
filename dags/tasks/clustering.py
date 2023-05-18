from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from tasks.files_util import save_files, load_files
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import datetime

# MLflow
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore")

def kmean_cluster(ti, **context) :
    df = load_files(['cleaned_fondue'])[0]

    selected = ['latitude', 'longitude']
    df_describe = df[selected]

    x = df_describe.values

    # Standardize variables with Standard Scaler
    scaler = StandardScaler()
    scaler.fit(x)

    cluster_data = scaler.fit_transform(x)

    #create a new experiment
    now = datetime.datetime.now()
    experiment_name = 'Airflow DAG triggered: {}'.format(now)
    try:
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        mlflow.delete_experiment(exp_id)
        exp_id = mlflow.create_experiment(name=experiment_name)

    max_cluster = 30
    labels = [None]

    # Run Experiment of this DAG trigger
    for k in range(1, max_cluster):
        with mlflow.start_run(experiment_id=exp_id, run_name=f"n_clusters = {k}") as run:
            kmeans = KMeans(n_clusters=k, random_state=2020)
            kmeans.fit(cluster_data)
            labels.append(kmeans.labels_)

            summation = sum(np.min(cdist(cluster_data, kmeans.cluster_centers_, 'euclidean'), axis=1))
            meandistortion = summation / cluster_data.shape[0]

            mlflow.log_param('n_clusters', k)
            mlflow.log_artifact('/opt/airflow/data/cleaned_fondue.csv')
            mlflow.log_metric('inertia', kmeans.inertia_)
            mlflow.log_metric('meandistortion', meandistortion)

            mlflow.sklearn.log_model(kmeans, 'model')

    chosen_k = 6

    # use the label from the best model
    df['cluster_label'] = labels[chosen_k]
    df.name = 'clustered_fondue'
    save_files([df])

        
if __name__ == '__main__':
    kmean_cluster(0)

