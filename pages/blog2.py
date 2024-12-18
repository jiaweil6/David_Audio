import streamlit as st
from sidebar import sidebar

st.markdown(
    """
    <style>
    /* Hide the entire Streamlit header */
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Blog Post #2: Binaural Beats")
st.write("""
Exploring the world of binaural beats and their impact on relaxation and focus.
""")

sidebar()
