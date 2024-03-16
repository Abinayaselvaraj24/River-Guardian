import folium
from folium import plugins
from streamlit_folium import folium_static
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from get_report import report
from geopy.geocoders import Nominatim

def change_font_color(element, text, font_color):
    st.markdown(f'<{element} style="color: {font_color};">{text}</{element}>', unsafe_allow_html=True)

class DashBoard:
    def __init__(self):
        pass

    def run(self):
        df = pd.read_csv('Data_Files/render.csv')

        change_font_color('h1',f"Total Plastic Count: {df['PRED_CT'].sum()}",'red')



        change_font_color('h1',"Dashboard",'black')

        def read_data(file_path):
            df = pd.read_csv(file_path)
            return df

        def get_city_name(row):
            geolocator = Nominatim(user_agent="plastic-heatmap")
            location = geolocator.reverse((row['latitude'], row['longitude']), exactly_one=True)
            return location.address.split(",")[-3].strip()



        df['CITY'] = df.apply(get_city_name, axis=1)

        sorted_df = df.groupby('CITY')['PRED_CT'].sum().reset_index().sort_values(by='PRED_CT', ascending=False)

        # Card-like structure for the highest count
        change_font_color('h3',f"Highest Count in City: {sorted_df['PRED_CT'][0]}",'red')



        # Plotly bar chart for the plastic count visualization using st.plotly_chart
        fig = go.Figure(go.Bar(x=sorted_df['CITY'], y=sorted_df['PRED_CT']))
        fig.update_layout(
            xaxis_title='City Location',
            yaxis_title='Plastic Count',
            xaxis_tickangle=45,
            plot_bgcolor='lightgray',
            xaxis=dict(tickfont=dict(color='black')),  # Set the x-axis tick color to black
            yaxis=dict(tickfont=dict(color='black')),
        )
        st.plotly_chart(fig)

        # Pie chart to show the distribution of plastic counts in different cities
        fig_pie = go.Figure(go.Pie(labels=sorted_df['CITY'], values=sorted_df['PRED_CT']))
        st.plotly_chart(fig_pie)

        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)

        st.header("Heatmap and Visualization")

        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # Scatter plot to visualize individual data points on the map
        scatter_data = df[['latitude', 'longitude', 'PRED_CT']]
        for _, row in scatter_data.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                popup=f"Plastic Number: {row['PRED_CT']}",
            ).add_to(m)

        # Heatmap for the plastic counts
        heat_data = df[['latitude', 'longitude', 'PRED_CT']].values.tolist()
        heat_map = plugins.HeatMap(heat_data, radius=15, min_opacity=0.5, max_val=df['PRED_CT'].max())
        m.add_child(heat_map)

        folium_static(m)
        report()












