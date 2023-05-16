from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
plt.style.use('ggplot')
import pandas as pd
from utils.files_util import save_files, load_files
from sklearn.preprocessing import StandardScaler
import numpy as np

import seaborn as sns
sns.set_style('whitegrid')

import folium

# size factor
factor = 3

# color procedure
def color_producer(state):
    if state == 'กำลังดำเนินการ':
        return 'blue'
    elif state == 'รอรับเรื่อง':
        return 'red'
    else:
        return 'green'

def compare_count_plot(data, column, title):
    # create a new font with Thai support
    font_path = '/opt/airflow/data/THSarabunChula-Regular.ttf'

    # data[column].value_counts()
    type_list = data[column].value_counts().index

    # set font
    sns.set(font_scale=2)
    sns.set(rc={'figure.figsize': (12, 8)})
    sns.set(font=fm.FontProperties(fname=font_path).get_name())

    # plot bar graph
    ax = sns.countplot(y=column, data=data, order=type_list)
    ax.bar_label(ax.containers[0])
    plt.title(title)
    plt.savefig('/opt/airflow/outputs/compare_count_plot.png')


def compare_pie_plot(data, column, title):
    # create a new font with Thai support
    font_path = '/opt/airflow/data/THSarabunChula-Regular.ttf'
    font_prop = fm.FontProperties(fname=font_path)

    # set the font as the default for Matplotlib
    plt.rcParams['font.family'] = font_prop.get_name()

    # set font
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15

    # data[column].value_counts()
    type_counts = data[column].value_counts()

    # Get four different grey colors
    cmap = plt.get_cmap('Greys')
    colors = list(cmap(np.linspace(0.45, 0.85, len(type_counts))))

    # Swap in a bright blue for the Lacrosse color
    colors[0] = 'dodgerblue'

    # plot pie chart
    plt.figure(figsize=(12, 8))
    patches, texts, pcts = plt.pie(
        type_counts, labels=type_counts.index,
        explode=[0.05, 0, 0], autopct='%1.1f%%', colors=colors,
        wedgeprops={'linewidth': 3.0, 'edgecolor': 'black'},
        textprops={'size': 'x-large'}, startangle=90
    )

    # Set the corresponding text label color to the wedge's face color
    for i, patch in enumerate(patches):
        texts[i].set_color(patch.get_facecolor())

    # style just the percent values
    plt.setp(pcts, color='white', fontweight='bold')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('/opt/airflow/outputs/compare_pie_plot.png')

def visualize_data(ti, **context) :
    df = load_files(['clustered_fondue'])[0]

    selected = ['latitude', 'longitude']
    df_describe = df[selected]

    x = df_describe.values

    # Standardize variables with Standard Scaler
    scaler = StandardScaler()
    scaler.fit(x)

    cluster_data = scaler.fit_transform(x)
    cluster_data

    #---------- Visualize dataset ----------#
    compare_count_plot(df, 'type', 'Number of Issues by Type')
    # TODO : Aggregate data by state, district, provice, to show type counts

    #---------- Visualize clusterd data ----------#
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    ax1.set_title('K-Means')
    ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], c=df["cluster_label"], cmap='rainbow')
    ax2.set_title("Original")
    ax2.scatter(cluster_data[:, 0], cluster_data[:, 1], cmap='rainbow')
    plt.savefig('/opt/airflow/outputs/clustered_data.png')

    #---------- Geospatial visualization Traffy By State ----------#
    # TODO: folium map of Traffy By State

    # Save traffy_map to html file
    # traffy_map.save("/opt/airflow/outputs/Map_by_State.html")

    #---------- Geospatial visualization Traffy By Size ----------#
    # TODO: folium map of Traffy By Size

    # Save traffy_map to html file
    # traffy_map.save("/opt/airflow/outputs/Map_by_Size.html")

    #---------- Geospatial visualization Low Income Heatmap ----------#
    from folium.plugins import HeatMap

    broke_df = load_files(['low_income'])[0]

    cols = ["X", "Y", "Lat", "Long"]
    for col in cols :
        broke_df[col] = broke_df[col].astype(float)
    broke_df = broke_df[broke_df["PROV_NAMT"] == "กรุงเทพมหานคร"]

    heat_df = broke_df[["Lat", "Long"]]
    # List comprehension to make out list of lists
    heat_data = [[row['Lat'], row['Long']] for index, row in heat_df.iterrows()]

    broke_map = folium.Map(location=[13.7563, 100.5668], zoom_start=12)
    folium.TileLayer('cartodbdark_matter').add_to(broke_map)

    # Display heatmap
    HeatMap(heat_data, radius=13, blur=15,).add_to(broke_map)
    broke_map.save("/opt/airflow/outputs/Low_Income_Heatmap.html")