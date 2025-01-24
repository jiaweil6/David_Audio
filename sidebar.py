import streamlit as st

def sidebar():
    st.sidebar.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/about.py", label="About Me", icon="🙋")
    st.sidebar.markdown("---")
    st.sidebar.page_link("pages/blog1.py", label="Convolution", icon="🎤")
    st.sidebar.page_link("pages/blog2.py", label="Impulse Response", icon="🎸")