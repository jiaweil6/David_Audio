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

st.page_link("blog1.py", label="Home", icon="🏠")
st.page_link("blog2.py", label="Page 1", icon="1️⃣")
st.page_link("blog3.py", label="Page 2", icon="2️⃣", disabled=True)
