# Import required libraries
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import folium_static
import pandas as pd
import base64

# Define the MarkerClusterPage class

def change_font_color(element, text, font_color):
    st.markdown(f'<{element} style="color: {font_color};">{text}</{element}>', unsafe_allow_html=True)

class MarkerClusterPage:
    def __init__(self):
        # Read the data from the CSV file
        self.df = pd.read_csv('Data_Files/render.csv')
        self.df['latitude'] = self.df['latitude'].astype(float)
        self.df['longitude'] = self.df['longitude'].astype(float)

    def render(self):
        # Set the title of the Streamlit app
        change_font_color('h1', "Marker Cluster",'black')

        # Create a Folium map centered on the mean latitude and longitude
        m = folium.Map(location=[self.df['latitude'].mean(), self.df['longitude'].mean()], zoom_start=12)

        # Create a marker cluster
        marker_cluster = plugins.MarkerCluster().add_to(m)

        # Add markers to the cluster based on the data in the DataFrame
        for idx, row in self.df.iterrows():
            popup = f"<br>Plastic Number: {row['PRED_CT']}"
            folium.Marker(location=[row['latitude'], row['longitude']], popup=popup).add_to(marker_cluster)

        # Display the Folium map using folium_static
        folium_static(m)

        # Add a button to download the data as a CSV file
        if st.button("Download Marker Data as CSV"):
            self.download_csv()

    def download_csv(self):
        # Prepare the data to be downloaded
        data_to_download = self.df[['latitude', 'longitude', 'PRED_CT']]

        # Create a link for downloading the data as a CSV file
        csv = data_to_download.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="marker_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)