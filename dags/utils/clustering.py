from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.files_util import save_files, load_files
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')

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

    n_clusters = 6

    kmeans = KMeans(n_clusters=n_clusters, random_state=2020)
    kmeans.fit(cluster_data)

    df['cluster_label'] = kmeans.labels_
    df.name = 'clustered_fondue'
    save_files([df])

    # Log mlflow
    mlflow.set_experiment('fondue')
    with mlflow.start_run() as run:
        mlflow.log_param('n_clusters', n_clusters)
        mlflow.log_metric('inertia', kmeans.inertia_)
        mlflow.sklearn.log_model(kmeans, 'model')

if __name__ == '__main__':
    kmean_cluster(0)

