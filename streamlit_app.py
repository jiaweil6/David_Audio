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

st.write("""
# Welcome to **DAVID AUDIO** 🎧  
Your destination for exploring the fascinating world of **audio synthesis** and **audio processing** in real-world applications!

While there’s no shortage of online resources about signals and systems, **practical, audio-focused guidance is scarce.**  
That’s where **theory meets practice**—helping you bridge the gap and bring your audio ideas to life.  

🎯 **What to explore?**  
- Dive into curated blog posts below  
- Learn more about me and my journey  

💡 **Want to contribute?**  
Email me at [jiaweil6@andrew.cmu.edu](mailto:jiaweil6@andrew.cmu.edu) and share your expertise to enrich the audio community!  
""")

st.latex(r''' 
    \int_{0}^{\infty} f(x) dx
''')


