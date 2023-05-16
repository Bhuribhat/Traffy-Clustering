import streamlit as st
import pandas as pd


def main():
    APP_TITLE = 'Traffy Fondue Clustering'
    APP_SUB_TITLE = 'Source: github'

    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    # df
    df = pd.read_csv("./data/teamchadchart.csv")
    st.write(df.shape)
    st.write(df.head(5))

    # Display HTML file
    st.title("HTML Files")
    html_file1 = open("./assets/Map_by_Size.html", "r", encoding="utf-8").read()
    st.components.v1.html(html_file1, width=500, height=500)

    html_file2 = open("./assets/Map_by_State.html", "r", encoding="utf-8").read()
    st.components.v1.html(html_file2, width=500, height=500)

    # # Display images
    # st.title("Images")
    # image1 = st.image("image1.jpg", caption="Image 1")
    # image2 = st.image("image2.jpg", caption="Image 2")


# >> streamlit run streamlit_app.py
if __name__ == '__main__':
    main()
