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


st.title("Blog Post #1: Analog Synthesizers")
st.write("""
Analog synthesizers were the foundation of electronic music. In this post, we explore their history,
unique sound characteristics, and how they continue to shape modern music production.
""")

sidebar()
