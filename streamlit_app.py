import streamlit as st
from sidebar import sidebar

sidebar()

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

sidebar_logo = "images/main_logo.png"
main_body_logo = "images/icon.png"

st.logo(sidebar_logo, size="large", icon_image=main_body_logo)


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg">
    <div class="nav-container">
        <div class="nav-content">
            <img src="static/logo.png" alt="Logo" class="nav-logo">
            <h1>DAVID AUDIO</h1>
            <a href="about" class="nav-button">About Me</a>
        </div>
    </div>
</nav>
""", unsafe_allow_html=True)

# Add some padding to prevent content from being hidden behind the navbar
st.markdown('<div style="margin-top: 100px;"></div>', unsafe_allow_html=True)

st.write("Welcome to David Audio, a space where we delve into music, soundscapes, and audio equipment. "
        "Choose a blog post below or learn more about me.")

# Add footer
st.markdown("""
<footer class="footer">
    <div class="footer-content">
        <p>Send an email to <a href="mailto:contribute@audiocommunity.com">contribute@audiocommunity.com</a> if you want to contribute to the audio community.</p>
    </div>
</footer>
""", unsafe_allow_html=True)


