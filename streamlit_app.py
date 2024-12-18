import streamlit as st
from sidebar import sidebar

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



st.title("David Audio")
st.write("Welcome to David Audio, a space where we delve into music, soundscapes, and audio equipment. "
        "Choose a blog post below or learn more about me.")

sidebar()
