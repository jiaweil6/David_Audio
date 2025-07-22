import streamlit as st
from sidebar import sidebar


st.set_page_config(
    page_title="IR",
    page_icon="🎸",
    layout="centered",
    initial_sidebar_state="auto",
)

def normalize_gain(signal, target_peak=0.9):
    """
    Normalize the gain of an audio signal to a target peak level.

    Parameters:
    - signal: np.ndarray, the input audio signal.
    - target_peak: float, the target peak level (default is 0.9).

    Returns:
    - normalized_signal: np.ndarray, the gain-normalized audio signal.
    """
    # Find the current peak of the signal
    current_peak = np.max(np.abs(signal))
    
    # Avoid division by zero
    if current_peak == 0:
        return signal
    
    # Calculate the normalization factor
    normalization_factor = target_peak / current_peak
    
    # Apply the normalization factor to the signal
    normalized_signal = signal * normalization_factor
    
    return normalized_signal

sidebar()

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
# Add some padding to prevent content from being hidden behind the navbar
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.markdown("<h2 class='title';'>Coming real soon!</h2>", unsafe_allow_html=True)


st.markdown('<div style="margin-top: 200px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")





