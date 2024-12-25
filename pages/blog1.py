import streamlit as st
from sidebar import sidebar
from streamlit_react_flow import react_flow
import numpy as np
from pydub import AudioSegment

sidebar()
main_body_logo = "images/icon.png"
st.logo(main_body_logo)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
# Add some padding to prevent content from being hidden behind the navbar
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.markdown("<h2>What is <u>Convolution</u>? <br> What does it mean in audio?</h2>", unsafe_allow_html=True)

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

subsection = '''Writer: <a href="about" target="_blank" class="author">David Liu</a><br>
Editor: <a href="about" target="_blank" class="author">David Liu</a><br>
Date: 2024-12-22
'''

st.markdown(subsection, unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Introduction")

st.write("""
Convolution is a special mathematical operation that combines two signals to procure a third signal. 
Probably heard it from Convolution-reverb and IR simulation. You don’t want to read through the entire Wikipedia page just to find out that you are still clueless about convolution.

Convolution of two signal $x$ and $h$ is written $x \\ast h$ where:
""")

st.latex(r"(x \ast h)(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) d\tau")

st.write("""
Don’t leave yet, I know you are probably clueless on what the heck this funky looking integral is. Exactly what I thought first time seeing it! 
Think of signal $x$ as the input signal, or the dry signal. And think of signal $h$ as a mystery box, or effect that allows signals to go through. 
""")

elements = [
    {
        "id": '1',
        "data": {"label": 'dry signal'},
        "type": "input",
        "position": {"x": 80, "y": 50},
        "sourcePosition": 'right',
    },
    {
        "id": '2',
        "data": {"label": 'effect'},
        "position": {"x": 280, "y": 50},
        "sourcePosition": 'right',
        "targetPosition": 'left',
        "style": {"background": "#66FCF1", "borderRadius": 40, "alignSelf": "center"}
    },
    {
        "id": '3',
        "data": {"label": "output"},
        "position": {"x": 480, "y": 50},
        "targetPosition": 'left',
        "sourcePosition": 'left',
    },
    {
        "id": 'e1-2',
        "source": '1',
        "target": '2',
        "animated": True
    },
    {
        "id": 'e2-3',
        "source": '2',
        "target": '3',
        "animated": True
    },
]


flowStyles = { "height": 90,"width":700 }
react_flow(
    "Convolution",
    elements=elements,
    flow_styles=flowStyles
)

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Intuition of Impulse Response")

st.write("""
    As said, signal $h$ just a effect we want to apply to the dry signal. We sometimes refer to this signal as the system or the impulse response.
    Mathematically, the impulse response is the output of the system when the input is an impulse signal.
""")

st.latex("x(t) = \delta(t)")
st.latex("(x \\ast h) (t) = y(t) = h(t)")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
    Imagine the impulse signal as a single spike of sound. Maybe a single drum hit, or a ballon pop which is the closest thing to a dirac delta function in real life.
    Now, imagine you played that in a concert hall, the sound will bounce off the walls and the ceiling, that short spike now with the concert hall reverberation is the impulse response.
    Although that is not how you retrieve the impulse response practically in real life, but it is a good intuition on the mathematical definition of impulse response.
""")

st.write("""
    Now that we captured the impulse response or the "effect", we can apply it to the dry signal.
    To obtain the output signal of the dry signal with the concert hall reverb, we will perform the convolution of the dry signal and the impulse response.
""")

st.latex("y(t) = (x \\ast h)(t)")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Practical Scenarios")

st.write("""
    In the digital world, we can't perform the convolution directly on the continuous signal due to the limited computer processing power.
    We will discretize the signal into samples and perform discrete convolution. Now the funky integral becomes a sum in discrete domain.
""")

st.latex("y[n] = \\sum_{k=0}^{N-1} x[k] h[n-k]")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
    Don't calculate your $y[n]$ just yet, imagine your $N$ is a billion where you have a super long signal. This summation would take forever! How can this be done in real-time?
    We the engineers got the trick to work around it. All signals have two domains, the time domain and the frequency domain.
    Convolution in time domain is actualy multiplication in the frequency domain! 
""")

st.latex("x[n] \\ast h[n] \longleftrightarrow X[e^{j\Omega}] \cdot H[e^{j\Omega}]")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
    How do we find the signal in frequency domain? This is rather technical but very very efficient.
    Fourier Transform, more specifically FFT (Fast Fourier Transform).
    Not going to scary you with big equations yet, it basically convert signal from it's time domain to it's frequency domain. 
    Vice versa, after multiplying two signals in the frequency domain, we perform the Inverse Fourier Fransform to get our time domain signal back which we could play it through a loud speaker.
""")

st.latex("x[n] \\longrightarrow X[e^{j\Omega}]")
st.latex("h[n] \\longrightarrow H[e^{j\Omega}]")
st.latex("Y[e^{j\Omega}] = X[e^{j\Omega}] \cdot H[e^{j\Omega}]")
st.latex("Y[e^{j\Omega}] \\longrightarrow y[n]")

st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

st.write("""
    Ready to test it out yourself?  
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Try it yourself!")

audio_value = st.audio_input("Record your voice here.")

if audio_value is not None:
    st.write("Converting your recorded audio to WAV in memory...")

    # 1. Retrieve raw bytes (WebM/Opus)
    webm_data = audio_value.getvalue()





st.markdown('<div style="margin-top: 200px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")





