import streamlit as st

# Set page configuration
st.set_page_config(page_title="David Audio", page_icon="🎧", layout="centered")

# Initialize session_state for page if not already set
if 'page' not in st.session_state:
    st.session_state.page = 'main'

def go_to(page_name: str):
    st.session_state.page = page_name

# Page navigation logic
current_page = st.session_state.page

# Hide Streamlit's default menu, footer, and GitHub icon
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stApp > header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Main page content
if current_page == 'main':
    st.title("David Audio")
    st.write("Welcome to David Audio, a space where we delve into music, soundscapes, and audio equipment. "
             "Choose a blog post below or learn more about me.")

    # Create columns for blog post previews
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.subheader("Post #1")
        st.write("A deep dive into analog synthesizers.")
        if st.button("Read More on Post #1"):
            go_to('blog1')

    with col2:
        st.subheader("Post #2")
        st.write("Exploring the world of binaural beats.")
        if st.button("Read More on Post #2"):
            go_to('blog2')

    with col3:
        st.subheader("Post #3")
        st.write("Headphones vs. Earbuds: which is better?")
        if st.button("Read More on Post #3"):
            go_to('blog3')

    st.markdown("---")
    if st.button("About Me"):
        go_to('about')

# Blog post and about pages
elif current_page == 'blog1':
    st.title("Blog Post #1: Analog Synthesizers")
    st.write("""
    Analog synthesizers were the foundation of electronic music. In this post, we explore their history,
    unique sound characteristics, and how they continue to shape modern music production.
    """)
    if st.button("Back to Main"):
        go_to('main')

elif current_page == 'blog2':
    st.title("Blog Post #2: Binaural Beats")
    st.write("""
    Binaural beats are an auditory illusion perceived when two slightly different frequencies are presented
    to each ear. We'll dive into the science, the claims, and whether they truly impact concentration and relaxation.
    """)
    if st.button("Back to Main"):
        go_to('main')

elif current_page == 'blog3':
    st.title("Blog Post #3: Headphones vs. Earbuds")
    st.write("""
    Headphones offer richer sound quality and noise isolation, while earbuds provide portability and convenience.
    In this post, we compare these two popular listening devices to help you decide what’s best for your needs.
    """)
    if st.button("Back to Main"):
        go_to('main')

elif current_page == 'about':
    st.title("About Me")
    st.write("""
    Hello! I'm David, an audio enthusiast, music producer, and sound engineer. 
    My passion for sound goes beyond listening — I love understanding how it's created, manipulated, and perceived.
    
    Through this blog, I share insights, reviews, and guidance to help you enrich your auditory experiences.
    Feel free to explore the various posts or get in touch to learn more!
    """)
    if st.button("Back to Main"):
        go_to('main')
