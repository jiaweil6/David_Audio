import streamlit as st
from sidebar import sidebar

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call this function at the start of your app
load_css()

st.markdown("<h1 style='color: #66FCF1;'>Your Title Here</h1>", unsafe_allow_html=True)
st.write("Welcome to David Audio, a space where we delve into music, soundscapes, and audio equipment. "
        "Choose a blog post below or learn more about me.")

sidebar()
