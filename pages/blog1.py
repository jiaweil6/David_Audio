import streamlit as st
from sidebar import sidebar

sidebar()
main_body_logo = "images/icon.png"
st.logo(main_body_logo)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
# Add some padding to prevent content from being hidden behind the navbar
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.markdown("<h2>What is <u>Convolution</u>? <br> What does it mean in audio?</h2>", unsafe_allow_html=True)

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

subsection = '''Writer: <a href="pages/about.py">David Liu</a><br>
Editor: <a href="pages/about.py">David Liu</a><br>
Date: 2024-12-22
'''

st.markdown(subsection, unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

text1 = """
    Convolution is a special mathematical operation that combines two signals to produce a third signals. Boring. 
"""

st.markdown(text1, unsafe_allow_html=True)

st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")




