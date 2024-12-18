import streamlit as st
from sidebar import sidebar

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)


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

sidebar()
