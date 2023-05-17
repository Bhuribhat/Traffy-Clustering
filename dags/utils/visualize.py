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

    #---------- Geospatial visualization Clustered Traffy ----------#
    # Get the cluster labels and their counts
    cluster_counts = df['cluster_label'].value_counts()

    # Generate a color palette based on the number of clusters
    n_clusters = len(cluster_counts)
    palette = sns.color_palette("viridis", n_clusters)

    # Color procedure
    def color_producer(cluster_label):
        return hex_colors[int(cluster_label)]

    def rgb_to_hex(rgb):
        r, g, b = rgb
        return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

    # Create a color dictionary mapping cluster labels to colors
    color_dict = dict(zip(cluster_counts.index, palette * 255))
    hex_colors = {k: rgb_to_hex(v) for k, v in color_dict.items()}

    def plot_cluster_by_problems(base_map):
        # Add Checkbox for filtering
        process_group  = folium.FeatureGroup(name="Ongoing").add_to(base_map)
        waiting_group  = folium.FeatureGroup(name="Upcoming").add_to(base_map)
        finished_group = folium.FeatureGroup(name="Complete").add_to(base_map)

        # Loop through the rows of the dataframe and add a marker for each location
        for index, row in df.iterrows():
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            label = int(row['cluster_label'])

            # Pop up information
            information = f"<b>Problem Type: {row['type']}</b><br><br>Cluster: {label}<br>Province: {row['province']}"
            # information += f"<br>District: {row['district']}<br>State: {row['state']}<br>Timestamp: {row['timestamp']}"

            iframe = folium.IFrame(information)
            popup = folium.Popup(iframe, min_width=300, max_width=300, min_height=140, max_height=170)

            # Add Group for filtering state
            if row['state'] == 'กำลังดำเนินการ':
                process_group.add_child(
                    folium.CircleMarker(
                        location=[lon, lat],
                        popup=popup,
                        tooltip=row['cluster_label'],
                        fill=True,
                        fill_color=color_producer(label),
                        color='black',
                        fill_opacity=0.7
                    )
                )
            elif row['state'] == 'รอรับเรื่อง':
                waiting_group.add_child(
                    folium.CircleMarker(
                        location=[lon, lat],
                        popup=popup,
                        tooltip=row['cluster_label'],
                        fill=True,
                        fill_color=color_producer(label),
                        color='black',
                        fill_opacity=0.7
                    )
                )
            else:
                finished_group.add_child(
                    folium.CircleMarker(
                        location=[lon, lat],
                        popup=popup,
                        tooltip=row['cluster_label'],
                        fill=True,
                        fill_color=color_producer(label),
                        color='black',
                        fill_opacity=0.7
                    )
                )

        # Add control layer
        folium.LayerControl().add_to(base_map)

    #---------- Geospatial visualization With Low Income Heatmap ----------#
    from folium.plugins import HeatMap

    broke_df = load_files(['low_income'])[0]

    cols = ["X", "Y", "Lat", "Long"]
    for col in cols :
        broke_df[col] = broke_df[col].astype(float)
    broke_df = broke_df[broke_df["PROV_NAMT"] == "กรุงเทพมหานคร"]

    heat_df = broke_df[["Lat", "Long"]]
    # List comprehension to make out list of lists
    heat_data = [[row['Lat'], row['Long']] for index, row in heat_df.iterrows()]

    broke_map = folium.Map(location=[13.7563, 100.5668], tiles="cartodbdark_matter", zoom_start=12)
    heatmap_group = folium.FeatureGroup(name="Heatmap").add_to(broke_map)
    heatmap_group.add_child(
        HeatMap(heat_data, radius=13, blur=15,)
    )

    plot_cluster_by_problems(broke_map)

    broke_map.save("/opt/airflow/outputs/Result_Map.html")