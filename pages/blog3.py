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

st.title("Blog Post #3: Headphones vs. Earbuds")
st.write("""
A comparison between headphones and earbuds to determine which is better for different scenarios.
""")

sidebar()
