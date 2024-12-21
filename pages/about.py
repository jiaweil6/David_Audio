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

st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">""", unsafe_allow_html=True)

st.write("""
Hi! I'm **David**, an engineer 👨‍🔬 and musician 🎸 currently pursuing a degree in Music and Technology as a sophomore at **Carnegie Mellon University** 🍈.

Signals and Systems 📡, a course taught at almost every university, finally bridged the gap between what I do every day in music and the underlying computations in audio processing 🔊.

While taking this course, I realized that it’s not always easy to find resources that connect theoretical concepts with their practical applications in audio. 

So, rather than calculating integrals here, I’d like to show you how these concepts are applied to audio.
I’m not saying the math isn’t important, but I’d like to make it more fun and engaging 😜.
""")

st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="contact-buttons">
    <a href="mailto:jiaweil6@andrew.cmu.edu" target="_blank">
        <button>
            <i class="bi bi-envelope"></i>
        </button>
    </a>
    <a href="https://www.linkedin.com/in/jiawei-liu-45110421a/" target="_blank">
        <button>
            <i class="bi bi-linkedin"></i>
        </button>
    </a>
    <a href="https://drive.google.com/file/d/1E08lU3L4KiYLvLIT6my5WD6OFdSnjYPg/view?usp=sharing" target="_blank">
        <button>
            <i class="bi bi-file-earmark-person"></i>
        </button>
    </a>
     <a href="https://www.jiaweismusic.com/" target="_blank">
        <button>
            <i class="bi bi-music-note-beamed"></i>
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")



