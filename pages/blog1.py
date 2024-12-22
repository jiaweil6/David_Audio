import streamlit as st
from sidebar import sidebar
from streamlit_react_flow import react_flow

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

st.write("""
Convolution is a special mathematical operation that combines two signals to procure a third signal. 
Probably heard it from Convolution-reverb and IR simulation. You don’t want to read through the entire Wikipedia page just to find out that you are still clueless about convolution.

Convolution of two signal $f$ and $h$ is written $f \\ast h$ where:
""")

st.latex(r"(f \ast h)(t) = \int_{-\infty}^{\infty} f(\tau) h(t - \tau) d\tau")

st.write("""
Don’t leave yet, I know you are probably clueless on what the heck this funky looking integral is. Exactly what I thought first time seeing it! 
Think of signal $f$ as the input signal, or the dry signal. And think of signal $h$ as a mystery box, or effect that allows signals to go through. 
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

st.write("""
    As said, signal $h$ just a effect we want to apply to the dry signal. We sometimes refer to this signal as the system or the impulse response.
    Mathematically, the impulse response is the output of the system when the input is an impulse signal.
""")

st.latex("x(t) = \delta(t)")
st.latex("y(t) = h(t)")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
    Imagine the impulse signal as a single spike of sound. Maybe a single drum hit, or a ballon pop which is the closest thing to a dirac delta function in real life.
    Now, imagine you played that in a concert hall, the sound will bounce off the walls and the ceiling, that short spike now with the concert hall reverberation is the impulse response.
    Although that is not how you retrieve the impulse response practically in real life, but it is a good intuition on the mathematical definition of impulse response.
""")

st.image("images/icon.png", use_container_width="always")





