import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.style.use('ggplot')

import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.font_manager.fontManager.addfont('THSarabunChula-Regular.ttf')
mpl.rc('font', family='TH Sarabun Chula')

# Load the Excel file into a Pandas dataframe
df = pd.read_csv('teamchadchart.csv')

# Drop some columns
drop_col = [
    'ticket_id', 'address', 'comment', 'photo', 'photo_after', 
    'star', 'count_reopen', 'last_activity', 'organization'
]
df.drop(drop_col, axis=1, inplace=True)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.date())

# split coordinate column into separate latitude and longitude columns
df[['latitude', 'longitude']] = df['coords'].str.split(',', expand=True)

# drop the original coordinate column
df.drop(['coords'], axis=1, inplace=True)
df.head()

# rename province in Bangkok
df['province'] = df['province'].apply(lambda x: "กรุงเทพมหานคร" if x == "จังหวัดกรุงเทพมหานคร" else x)

# split 'type' in each row to list datatype
df['type'] = df['type'].apply(lambda x: x.strip('{}').split(','))

# add new column to count 'type' length
df['type_count'] = df['type'].apply(lambda x: len(x))
df.head()

from scipy import stats

# convert coord column to float
df = df.astype({"latitude": float, "longitude": float})

# Calculate z-scores for latitude and longitude columns
z_lat  = np.abs(stats.zscore(df['latitude']))
z_long = np.abs(stats.zscore(df['longitude']))

# Define threshold value
threshold = 3

# Remove outliers from the dataframe
df = df[(z_lat < threshold) & (z_long < threshold)]
print(f"There are {df.shape[0]} rows after removing outliners.")

# explode list of type to multiple rows
df_exploded = df.explode('type')
df_exploded.drop('type_count', axis=1, inplace=True)
df_exploded.head()

print(f"There are {df.shape[0]} rows in df dataframe.")
print(f"There are {df_exploded.shape[0]} rows in df_exploded dataframe.")

df_exploded.info()

# number of missing values
null_counts = df_exploded.isnull().sum()
print(f"\nNumber of null values in each column after imputing:\n{null_counts}")

# before cleaning empty string
print(f"There are {df_exploded.shape[0]} rows before cleaning.")

# after cleaning empty string
df_exploded['type'] = df_exploded['type'].str.strip()
df_exploded.drop(df_exploded[df_exploded['type'] == ''].index, inplace=True)
print(f"There are {df_exploded.shape[0]} rows after cleaning.")

def compare_count_plot(data, column, title):
    # create a new font with Thai support
    font_path = 'THSarabunChula-Regular.ttf'

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
    plt.show()


def compare_pie_plot(data, column, title):
    # create a new font with Thai support
    font_path = 'THSarabunChula-Regular.ttf'
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
    plt.show()

compare_count_plot(df_exploded, 'type', 'Number of Issues by Type')

compare_count_plot(df, 'district', 'District')

compare_count_plot(df, 'type_count', 'Problems Count')

compare_pie_plot(df, 'state', None)

df['province'].value_counts()

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

# Create a base map centered on a Thailand location
traffy_map = folium.Map(location=[13.7563, 100.5018], tiles="Stamen Terrain", zoom_start=12)

# Add Checkbox for filtering
process_group  = folium.FeatureGroup(name="Ongoing").add_to(traffy_map)
waiting_group  = folium.FeatureGroup(name="Upcoming").add_to(traffy_map)
finished_group = folium.FeatureGroup(name="Complete").add_to(traffy_map)

# Loop through the rows of the dataframe and add a marker for each location
for index, row in df.iterrows():
    lat = float(row['latitude'])
    lon = float(row['longitude'])
    count = int(row['type_count'])

    # Pop up information
    information = f"<b>Problem Count: {count}</b><br><br>Province: {row['province']}<br>District: {row['district']}"
    information += f"<br>State: {row['state']}<br>Problem: {row['type']}<br>Timestamp: {row['timestamp']}"

    iframe = folium.IFrame(information)
    popup = folium.Popup(iframe, min_width=300, max_width=300, min_height=150, max_height=180)

    # Add Group for filtering state
    if row['state'] == 'กำลังดำเนินการ':
        process_group.add_child(
            folium.CircleMarker(
                location=[lon, lat],
                radius=count * factor,
                popup=popup,
                tooltip=row['district'],
                fill=True,
                fill_color=color_producer(row['state']),
                color='black',
                fill_opacity=0.7
            )
        )
    elif row['state'] == 'รอรับเรื่อง':
        waiting_group.add_child(
            folium.CircleMarker(
                location=[lon, lat],
                radius=count * factor,
                popup=popup,
                tooltip=row['district'],
                fill=True,
                fill_color=color_producer(row['state']),
                color='black',
                fill_opacity=0.7
            )
        )
    else:
        finished_group.add_child(
            folium.CircleMarker(
                location=[lon, lat],
                radius=count * factor,
                popup=popup,
                tooltip=row['district'],
                fill=True,
                fill_color=color_producer(row['state']),
                color='black',
                fill_opacity=0.7
            )
        )

# Add control layer
folium.LayerControl().add_to(traffy_map)

# Save traffy_map to html file
# traffy_map.save("Map_by_State.html")
traffy_map

import folium

# size factor
factor = 3

# color procedure
def color_producer(count):
    if count == 1:
        return 'green'
    elif count == 2:
        return 'blue'
    elif count == 3:
        return 'orange'
    else:
        return 'red'

# Create a base map centered on a Thailand location
traffy_map = folium.Map(location=[13.7563, 100.5018], tiles="Stamen Terrain", zoom_start=12)

# Add Checkbox for filtering
group_1 = folium.FeatureGroup(name="Type Count 1").add_to(traffy_map)
group_2 = folium.FeatureGroup(name="Type Count 2").add_to(traffy_map)
group_3 = folium.FeatureGroup(name="Type Count 3").add_to(traffy_map)
group_4 = folium.FeatureGroup(name="Type Count 4").add_to(traffy_map)
group_5 = folium.FeatureGroup(name="Type Count 5").add_to(traffy_map)

# Loop through the rows of the dataframe and add a marker for each location
for index, row in df.iterrows():
    lat = float(row['latitude'])
    lon = float(row['longitude'])
    count = int(row['type_count'])

    # Pop up information
    information = f"<b>Problem Count: {count}</b><br><br>Province: {row['province']}<br>District: {row['district']}"
    information += f"<br>State: {row['state']}<br>Problem: {row['type']}<br>Timestamp: {row['timestamp']}"

    iframe = folium.IFrame(information)
    popup = folium.Popup(iframe, min_width=300, max_width=300, min_height=150, max_height=180)

    # Add Group for filtering count
    if count == 1:
        group_1.add_child(
            folium.CircleMarker(
                location=[lon, lat],
                radius=count * factor,
                popup=popup,
                tooltip=row['district'],
                fill=True,
                fill_color=color_producer(count),
                color='black',
                fill_opacity=0.7
            )
        )
    elif count == 2:
        group_2.add_child(
            folium.CircleMarker(
                location=[lon, lat],
                radius=count * factor,
                popup=popup,
                tooltip=row['district'],
                fill=True,
                fill_color=color_producer(count),
                color='black',
                fill_opacity=0.7
            )
        )
    elif count == 3:
        group_3.add_child(
            folium.CircleMarker(
                location=[lon, lat],
                radius=count * factor,
                popup=popup,
                tooltip=row['district'],
                fill=True,
                fill_color=color_producer(count),
                color='black',
                fill_opacity=0.7
            )
        )
    elif count == 4:
        group_4.add_child(
            folium.CircleMarker(
                location=[lon, lat],
                radius=count * factor,
                popup=popup,
                tooltip=row['district'],
                fill=True,
                fill_color=color_producer(count),
                color='black',
                fill_opacity=0.7
            )
        )
    else:
        group_5.add_child(
            folium.CircleMarker(
                location=[lon, lat],
                radius=count * factor,
                popup=popup,
                tooltip=row['district'],
                fill=True,
                fill_color=color_producer(count),
                color='black',
                fill_opacity=0.7
            )
        )

# Add control layer
folium.LayerControl().add_to(traffy_map)

# Save traffy_map to html file
# traffy_map.save("Map_by_Size.html")
traffy_map

from sklearn.preprocessing import StandardScaler

selected = ['latitude', 'longitude']
df_describe = df[selected]

x = df_describe.values

# Standardize variables with Standard Scaler
scaler = StandardScaler()
scaler.fit(x)

cluster_data = scaler.fit_transform(x)
cluster_data

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Initialize the plot with the specified dimensions
fig = plt.figure(figsize=(12, 8))

# elbow method, to determine the optimal number of clusters
meandistortions = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=2020)
    kmeans.fit(cluster_data)
    summation = sum(np.min(cdist(cluster_data, kmeans.cluster_centers_, 'euclidean'), axis=1))
    meandistortions.append(summation / cluster_data.shape[0])

plt.plot(range(1, 10), meandistortions, 'rx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=2, random_state=2020)
kmeans.fit(cluster_data)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

ax1.set_title('K-Means')
ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], c=kmeans.labels_, cmap='rainbow')

ax2.set_title("Original")
ax2.scatter(cluster_data[:, 0], cluster_data[:, 1], cmap='rainbow')
plt.show()

# inverse transform to get original centroids back
original_data = scaler.inverse_transform(kmeans.cluster_centers_)

# get unique size for each clusters
(unique, counts) = np.unique(kmeans.labels_, return_counts=True)

print(f"Normalize data:\n{'-' * 30}")
for idx, (x, y) in enumerate(kmeans.cluster_centers_):
    print(f"centroid {idx + 1} = ({x:5.2f}, {y:5.2f}), size = {counts[idx]}")

print(f"\nOriginal data:\n{'-' * 30}")
for idx, (x, y) in enumerate(original_data):
    print(f"centroid {idx + 1} = ({x:.2f}, {y:.2f})")

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(12, 8))

# Colors uses a color map, which will produce an array of colors based on the
# number of labels there are. We use set(kmeans_labels) to get the unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(kmeans.labels_))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
for k, col in zip(range(len(kmeans.cluster_centers_)), colors):

    # Create a list of all data points, where the data points that are in the cluster 
    # (ex. cluster 0) are labeled as true, else they are labeled as false.
    my_members = (kmeans.labels_ == k)
    
    # Define the centroid, or cluster center.
    cluster_center = kmeans.cluster_centers_[k]
    
    # Plots the datapoints with color col.
    ax.plot(cluster_data[my_members, 0], cluster_data[my_members, 1], 'o', markerfacecolor=col, markersize=10)
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=20)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

from sklearn.metrics.pairwise import haversine_distances

# Handling Location
points_in_radians = df[['latitude', 'longitude']].apply(np.radians).values
distances_in_km = haversine_distances(points_in_radians) * 6371

from sklearn.cluster import DBSCAN

# distance_matrix = rating_distances + distances_in_km
distance_matrix = distances_in_km

clustering = DBSCAN(metric='precomputed', eps=1, min_samples=3)
clustering.fit(distance_matrix)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

ax1.set_title('DBSCAN')
ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], c=clustering.labels_, cmap='rainbow')

ax2.set_title("Original")
ax2.scatter(cluster_data[:, 0], cluster_data[:, 1], cmap='rainbow')
plt.show()

import folium
from folium import plugins
from folium.plugins import HeatMap


broke_df = pd.read_csv("broke.csv")

cols = ["X", "Y", "Lat", "Long"]
for col in cols :
  broke_df[col] = broke_df[col].astype(float)
broke_df = broke_df[broke_df["PROV_NAMT"] == "กรุงเทพมหานคร"]

broke_df.head(4)

heat_df = broke_df[["Lat", "Long"]]

# List comprehension to make out list of lists
heat_data = [[row['Lat'], row['Long']] for index, row in heat_df.iterrows()]

broke_map = folium.Map(location=[13.7563, 100.5668], zoom_start=12)
folium.TileLayer('cartodbdark_matter').add_to(broke_map)

# Display heatmap
HeatMap(heat_data, radius=13, blur=15,).add_to(broke_map)
broke_map

import geopandas as gpd

# Create a GeoDataFrame from the broken locations
geometry = gpd.points_from_xy(broke_df['Long'], broke_df['Lat'])
broke_gdf = gpd.GeoDataFrame(broke_df, geometry=geometry)

# Create a KDE plot using the GeoDataFrame
fig, ax = plt.subplots(figsize=(12, 8))
broke_gdf.plot(ax=ax, cmap='hot', alpha=0.7)

# Set the axis labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Broken Locations Heatmap')

# Show the plot
plt.show()

import geopandas as gpd

# Create a GeoDataFrame from the broken locations
geometry = gpd.points_from_xy(broke_df['Long'], broke_df['Lat'])
broke_gdf = gpd.GeoDataFrame(broke_df, geometry=geometry)

# Create a KDE plot using the GeoDataFrame
fig, ax = plt.subplots(figsize=(12, 8))
sns.kdeplot(
    data=broke_df, x='Long', y='Lat', cmap='hot', 
    alpha=0.7, shade=True, shade_lowest=True
)

# Set the axis labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Broken Locations Heatmap')

# Show the plot
plt.show()
