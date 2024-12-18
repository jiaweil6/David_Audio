import streamlit as st

def sidebar():
    st.sidebar.page_link("streamlit_app.py", label="Home", icon="🎧")
    st.sidebar.page_link("pages/blog1.py", label="Blog 1", icon="🎧")
    st.sidebar.page_link("pages/blog2.py", label="Blog 2", icon="🎧")
    st.sidebar.page_link("pages/blog3.py", label="Blog 3", icon="🎧")