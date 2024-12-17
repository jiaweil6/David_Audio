import streamlit as st

def about(go_to):
    st.title("About Me")
    st.write("""
    Hello! I'm David, an audio enthusiast, music producer, and sound engineer. 
    My passion for sound goes beyond listening — I love understanding how it's created, manipulated, and perceived.
    
    Through this blog, I share insights, reviews, and guidance to help you enrich your auditory experiences.
    Feel free to explore the various posts or get in touch to learn more!
    """)
    if st.button("Back to Main"):
        go_to('main')
