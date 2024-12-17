import streamlit as st

def blog1(go_to):
    st.title("Blog Post #1: Analog Synthesizers")
    st.write("""
    Analog synthesizers were the foundation of electronic music. In this post, we explore their history,
    unique sound characteristics, and how they continue to shape modern music production.
    """)
    if st.button("Back to Main"):
        go_to('main')
