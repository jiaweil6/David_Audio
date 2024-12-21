import streamlit as st
from sidebar import sidebar

sidebar()

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

sidebar_logo = "images/main_logo.png"
main_body_logo = "images/icon.png"

st.logo(sidebar_logo, size="large", icon_image=main_body_logo)

# Add some padding to prevent content from being hidden behind the navbar
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.html("<h1 style='color:#C5C6C7;'>Welcome to</h1> <h1>DAVID AUDIO 🎧</h1>")

st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

st.write("""
A space where we break down **audio synthesis** and **audio processing** into easy-to-understand demonstrations,  
showing how theory directly applies to real-world audio applications.  

While there’s a wealth of resources on signals and systems, audio-specific guides are hard to come by. 

Here, we focus on bridging the gap—offering clear examples that connect the dots between theory and its practical use in audio.   

💌 **Contribute**  
Got insights to share? Email me at [jiaweil6@andrew.cmu.edu](mailto:jiaweil6@andrew.cmu.edu) to help grow this audio community.  
""")



