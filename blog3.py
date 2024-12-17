import streamlit as st

def blog3(go_to):
    st.title("Blog Post #3: Headphones vs. Earbuds")
    st.write("""
    Headphones offer richer sound quality and noise isolation, while earbuds provide portability and convenience.
    In this post, we compare these two popular listening devices to help you decide what’s best for your needs.
    """)
    if st.button("Back to Main"):
        go_to('main')
