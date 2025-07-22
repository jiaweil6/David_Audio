import streamlit as st


def sidebar():
    with st.sidebar:
        st.sidebar.page_link("streamlit_app.py", label="Home", icon="🏠")
        st.sidebar.page_link("pages/about.py", label="About Me", icon="🙋")
        st.sidebar.markdown("---")
        st.sidebar.page_link("pages/convolution.py", label="Convolution", icon="🎤")
        st.sidebar.page_link("pages/phase_vocoder.py", label="Phase Vocoder", icon="📽️")
        st.sidebar.page_link("pages/impulse_response.py", label="Impulse Response", icon="🎸")