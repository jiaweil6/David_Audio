import streamlit as st

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

st.title("David Audio")
st.write("Welcome to David Audio, a space where we delve into music, soundscapes, and audio equipment. "
        "Choose a blog post below or learn more about me.")

# Custom page links with names
st.markdown("[Blog Post 1](pages/blog1.py)")
st.markdown("[Blog Post 2](pages/blog2.py)")
st.markdown("[Blog Post 3](pages/blog3.py)")
