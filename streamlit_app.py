import streamlit as st
from blog1 import blog1
from blog2 import blog2
from blog3 import blog3
from about import about

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

# Page routing
elif current_page == 'blog1':
    blog1(go_to)
elif current_page == 'blog2':
    blog2(go_to)
elif current_page == 'blog3':
    blog3(go_to)
elif current_page == 'about':
    about(go_to)
