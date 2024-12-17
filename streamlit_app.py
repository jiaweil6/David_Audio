import streamlit as st
from blog1 import blog1_page
from blog2 import blog2_page
from blog3 import blog3_page
from about import about_page

# Set page configuration
st.set_page_config(page_title="David Audio", page_icon="🎧", layout="centered")

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