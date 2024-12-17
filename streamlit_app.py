import streamlit as st
from blog1 import blog1_page
from blog2 import blog2_page
from blog3 import blog3_page
from about import about_page

# Set page configuration
st.set_page_config(
    page_title="David Audio", 
    page_icon="🎧", 
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS to hide the entire header
st.markdown(
    """
    <style>
    /* Hide the entire Streamlit header */
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS to style the sidebar
st.markdown(
    """
    <style>
    /* Style the sidebar */
    .css-1d391kg {
        background-color: #f0f2f6; /* Light background color */
        padding: 20px; /* Add padding */
    }
    .css-1d391kg h2 {
        color: #333333; /* Darker text color for headings */
        font-size: 24px; /* Larger font size for headings */
    }
    .css-1d391kg .stRadio {
        color: #333333; /* Darker text color for radio buttons */
    }
    .css-1d391kg .stRadio div {
        font-size: 18px; /* Larger font size for radio button labels */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ("Main", "Blog 1", "Blog 2", "Blog 3", "About"))

# Page routing
if page == "Main":
    st.title("David Audio")
    st.write("Welcome to David Audio, a space where we delve into music, soundscapes, and audio equipment. "
             "Choose a blog post below or learn more about me.")
elif page == "Blog 1":
    blog1_page()
elif page == "Blog 2":
    blog2_page()
elif page == "Blog 3":
    blog3_page()
elif page == "About":
    about_page()
