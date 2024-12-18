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

with st.sidebar:
    st.page_link("pages/blog1.py", label="Blog 1", icon="🎧", disabled=True)
    st.page_link("pages/blog2.py", label="Blog 2", icon="🎧", disabled=True)
    st.page_link("pages/blog3.py", label="Blog 3", icon="🎧", disabled=True)
