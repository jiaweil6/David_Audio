import streamlit as st
from sidebar import sidebar

sidebar()
sidebar_logo = "images/main_logo.png"
main_body_logo = "images/icon.png"

st.logo(sidebar_logo, size="large", icon_image=main_body_logo)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
# Add some padding to prevent content from being hidden behind the navbar
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.html("<h1> <span style='color:#F8F8FF;'>About Me</span></h1>")

# st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)

st.write("""
Hi! I'm David, an engineer and musician. I'm currently a sophomore at Carnegie Mellon University, pursuing a degree in Music and Technology. 
Signals and Systems, a course taught in almost every university, finally linked the bridge between what I've been doing everyday in music and what it is really under the computer.
While taking this course, I found it is not easy to find resources that connect the dots between theory and its practical use in audio.
So rather than calculating the integrals here, I'd like to show you how it is applied to audio. I'm not saying the math is not important, but I'd like to make math more fun and interesting.
""")

st.image("images/icon.png", use_container_width="always")



