import streamlit as st
from main_page import main_page
from blog1_page import blog1_page
from blog2_page import blog2_page
from blog3_page import blog3_page
from about_page import about_page

# Set page configuration
st.set_page_config(page_title="David Audio", page_icon="🎧", layout="centered")

# Initialize session_state for page if not already set
if 'page' not in st.session_state:
    st.session_state.page = 'main'

def go_to(page_name: str):
    st.session_state.page = page_name

# Page navigation logic
current_page = st.session_state.page

# Hide Streamlit's default menu, footer, and GitHub icon
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stApp > header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Page routing
if current_page == 'main':
    main_page(go_to)
elif current_page == 'blog1':
    blog1_page(go_to)
elif current_page == 'blog2':
    blog2_page(go_to)
elif current_page == 'blog3':
    blog3_page(go_to)
elif current_page == 'about':
    about_page(go_to)
