import streamlit as st
from sidebar import sidebar

sidebar()

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

sidebar_logo = "images/main_logo.png"
main_body_logo = "images/icon.png"

st.logo(sidebar_logo, size="large", icon_image=main_body_logo)

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


