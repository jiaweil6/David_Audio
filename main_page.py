import streamlit as st

def main_page(go_to):
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
