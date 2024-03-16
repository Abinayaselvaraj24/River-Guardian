# Import required libraries
import warnings
warnings.filterwarnings('ignore')
from streamlit_option_menu import option_menu
import base64
import streamlit as st
from predict import Prediction  # Importing a module named 'Prediction'
from marker_cluster import MarkerClusterPage
from dashboard import DashBoard
import requests
from io import BytesIO
from realtime import realtime

# Define a function to set a background image
def set_background(image_url):
    # Get the image from the URL
    response = requests.get(image_url)
    image_bytes = BytesIO(response.content)
    bin_str = base64.b64encode(image_bytes.read()).decode()

    # Apply the background image using CSS
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''

# Define a function to get the base64 encoded data of a binary file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Define a function to apply local CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Use the 'option_menu' component to create a navigation menu
selected = option_menu(
    menu_title=None,
    options=['Home', 'Predict', 'RealTime', 'Locator', "Insights"],
    icons=['house-door-fill', 'upc-scan', 'eye-fill', 'zoom-in', "clipboard-data-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# main function of your Streamlit web application
def main():
    # Apply local CSS styles defined in "assets/styles.css"
    local_css("assets/styles.css")

    # Check the value of the selected menu option
    if selected == "Home":
        # Display images for the Home section
        st.image('assets/RiverGuard.png')
        st.image("assets/home2.png")

    elif selected == "Predict":
        # If "Predict" option is selected, create an instance of the Prediction class and call the prediction() method
        p = Prediction()
        p.prediction()

    elif selected == "Locator":
        try:
            # If "Locator" option is selected, create an instance of the MarkerClusterPage class and render it
            m = MarkerClusterPage()
            m.render()
        except:
            st.warning("OOPS :) Something went wrong...!")

    elif selected == "Insights":
        try:
            # If "Insights" option is selected, create an instance of the DashBoard class and run it
            m = DashBoard()
            m.run()
        except:
            st.warning("OOPS :) Something went wrong...!")

    elif selected == "RealTime":
        # If "RealTime" option is selected, call the realtime() function
        realtime()

# Check if the script is being run directly
if __name__ == '__main__':
    # Call the main function to start the Streamlit web application
    main()



