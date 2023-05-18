import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import seaborn as sns
sns.set_style('whitegrid')

import matplotlib as mpl
mpl.font_manager.fontManager.addfont('./data/THSarabunChula-Regular.ttf')
mpl.rc('font', family='TH Sarabun Chula')


def selectbox_without_default(label, options):
    st.sidebar.markdown(
        "<h1 style='text-align: center; margin-bottom: 30px;'>Navigation Bar</h1>", 
        unsafe_allow_html=True
    )
    options = [''] + options
    format_func = lambda x: 'Select an option' if x == '' else x
    return st.sidebar.selectbox(label, options, format_func=format_func)


def choose_sidebar_vis():
    options = ['Geospatial', 'Heatmap']
    return st.sidebar.radio('Type', options)


def compare_count_plot(data, column, title):
    font_path = './data/THSarabunChula-Regular.ttf'
    type_list = data[column].value_counts().index

    # set font
    sns.set(font_scale=2)
    sns.set(rc={'figure.figsize': (12, 12)})
    sns.set(font=fm.FontProperties(fname=font_path).get_name())

    # plot bar graph
    ax = sns.countplot(y=column, data=data, order=type_list)
    ax.bar_label(ax.containers[0])
    plt.title(title)
    plt.tight_layout()

    # Return the plot instead of calling plt.show()
    return plt


def clean_data_explode_type(data):
    # Drop some columns
    drop_col = [
        'ticket_id', 'description', 'after_photo', 'photo_url', 
        'star', 'count_reopen', 'last_activity', 'org'
    ]
    data.drop(drop_col, axis=1, inplace=True)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp']).apply(lambda x: x.date())

    # split coordinate column into separate latitude and longitude columns
    data[['latitude', 'longitude']] = data['coords'].apply(lambda x: x.strip('\"[]\"')).str.split(', ', expand=True)
    data['latitude'] = data['latitude'].apply(lambda x: x.strip("\'"))
    data['longitude'] = data['longitude'].apply(lambda x: x.strip("\'"))

    # drop the original coordinate column
    data.drop(['coords'], axis=1, inplace=True)
    data['province'] = data['address'].str.split(' ').str[-3]

    # rename province in Bangkok
    data['province'] = data['province'].apply(lambda x: "กรุงเทพมหานคร" if x == "จังหวัดกรุงเทพมหานคร" else x)
    data['district'] = data['address'].str.split(' ').str[-4]

    # split 'type' in each row to list datatype
    data['type'] = data['problem_type_abdul'].apply(lambda x: x.strip('\"[]').replace("\'", "").split(", "))

    # add new column to count 'type' length
    data['type_count'] = data['type'].apply(lambda x: len(x))

    # convert coord column to float
    data = data.astype({"latitude": float, "longitude": float})

    # explode list of type to multiple rows
    df_exploded = data.explode('type')
    df_exploded.drop('type_count', axis=1, inplace=True)

    # cleaning empty string
    df_exploded['type'] = df_exploded['type'].str.strip("\'\"")
    df_exploded.drop(df_exploded[df_exploded['type'] == ''].index, inplace=True)
    return df_exploded


def main():
    APP_TITLE = 'Traffy Fondue Clustering'
    APP_SUB_TITLE = 'Source: https://share.traffy.in.th/teamchadchart'

    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    # Read Dataframe
    df = pd.read_csv("./data/data_cleaned_by_length.csv")
    df_explode = pd.read_csv("./data/fondue.csv")
    df_explode = clean_data_explode_type(df_explode)
    df.drop(['problem_type_abdul', 'note'], axis=1, inplace=True)

    # Display Visualization images
    options = ['type', 'district', 'state', 'type_count']
    data_choice = selectbox_without_default("Choose a column", options)
    radio = choose_sidebar_vis()

    if data_choice is not '':
        st.subheader("Data Visualization")
        st.write('Selected option:', data_choice)
        if data_choice != 'type':
            st.pyplot(compare_count_plot(df, data_choice, data_choice.upper()))
        else:
            st.pyplot(compare_count_plot(df_explode, data_choice, 'Number of Issues by Type'))

    if radio == 'Geospatial':
        st.write(df.head(5))

        # Display HTML file
        st.subheader("Geospatial Visualization")
        html_file1 = open("./outputs/Map_by_Size.html", "r", encoding="utf-8").read()
        st.components.v1.html(html_file1, width=700, height=450)

    else:
        # Clustering Model
        st.subheader("K-means Clustering")
        st.image("./outputs/clustered_data.png", caption="clustered_data")

        # Heatmap
        st.subheader("Heatmap Visualization")
        st.caption("Source: https://gdcatalog.nha.co.th/dataset/dataset11_01")

        df_broke = pd.read_csv("./data/low_income.csv")
        st.write(df_broke.head(5))

        html_file2 = open("./outputs/Result_Map.html", "r", encoding="utf-8").read()
        st.components.v1.html(html_file2, width=700, height=450)


if __name__ == '__main__':
    main()
