import streamlit as st
from sidebar import sidebar

sidebar()
sidebar_logo = "images/main_logo.png"
main_body_logo = "images/icon.png"

st.logo(sidebar_logo, size="large", icon_image=main_body_logo)

st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">""", unsafe_allow_html=True)

st.write("""
Hi! I'm David, an engineer 👨‍🔬 and musician 🎸 currently pursuing a degree in Music and Technology as a sophomore at Carnegie Mellon University 🍈.
Signals and Systems 📡, a course taught at almost every university, finally bridged the gap between what I do every day in music and the underlying computations in audio processing 🔊.
While taking this course, I realized that it’s not always easy to find resources that connect theoretical concepts with their practical applications in audio. 
So, rather than calculating integrals here, I’d like to show you how these concepts are applied to audio.
I’m not saying the math isn’t important, but I’d like to make it more fun and engaging 😜.
""")

# Add buttons with icons for email, LinkedIn, and CV
st.markdown("""
<div style="display: flex; justify-content: space-around; margin-top: 20px;">
    <a href="mailto:your-email@example.com" target="_blank">
        <button style="padding: 10px 20px; font-size: 16px;">
            <i class="fa fa-envelope"></i>
        </button>
    </a>
    <a href="https://www.linkedin.com/in/your-linkedin-profile" target="_blank">
        <button style="padding: 10px 20px; font-size: 16px;">
            <i class="fa-brands fa-linkedin"></i>
        </button>
    </a>
    <a href="path/to/your-cv.pdf" target="_blank">
        <button style="padding: 10px 20px; font-size: 16px;">
            <i class="fa-solid fa-file"></i>
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")



