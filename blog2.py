import streamlit as st

def blog2(go_to):
    st.title("Blog Post #2: Binaural Beats")
    st.write("""
    Binaural beats are an auditory illusion perceived when two slightly different frequencies are presented
    to each ear. We'll dive into the science, the claims, and whether they truly impact concentration and relaxation.
    """)
    if st.button("Back to Main"):
        go_to('main')
