import streamlit as st
from sidebar import sidebar
from streamlit_react_flow import react_flow
import numpy as np
import pandas as pd
import altair as alt
import io
import soundfile as sf

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

st.image("images/flow-chart.png", use_container_width="always")

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
    Convolution in time domain is actualy multiplication in the frequency domain! Much simpler now, but how do we find the signal in it's frequency domain?
""")

st.latex("x[n] \\ast h[n] \longleftrightarrow X[e^{j\Omega}] \cdot H[e^{j\Omega}]")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
    This is rather technical but very very efficient.
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
st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
# ------------------
# AUDIO INPUT
# ------------------
# (Assumes a custom widget that returns an UploadedFile-like object.)
audio_value = st.audio_input("Record your beautiful voice here.")

# We'll assume both audio inputs and IR are at 44100 Hz
sample_rate = 44100

# ------------------
# READ IMPULSE RESPONSE
# ------------------
IR_file = "audio/reverb.wav"
IR_data, ir_sample_rate = sf.read(IR_file)

# If IR is stereo, just grab the left channel, or handle as you wish
if IR_data.ndim == 2:
    IR_data = IR_data[:, 0]

# (Optional) If IR is at a different sample rate, consider resampling or
# at least acknowledging the difference. For simplicity, assume they're the same.
# For correctness:
#   if ir_sample_rate != sample_rate:
#       IR_data = librosa.resample(IR_data, orig_sr=ir_sample_rate, target_sr=sample_rate)

# ------------------
# FREQUENCY ANALYSIS OF IR
# ------------------
# Take the FFT of the IR
IR_frequency_domain = np.fft.fft(IR_data)
IR_frequency_bins = np.fft.fftfreq(len(IR_data), d=1.0 / sample_rate)
IR_amplitude_spectrum = np.abs(IR_frequency_domain)

# Keep only 20 Hz to 20 kHz
mask = (IR_frequency_bins >= 20) & (IR_frequency_bins <= 20000)
IR_freqs = IR_frequency_bins[mask]
IR_amps = IR_amplitude_spectrum[mask]

# Create a DataFrame for Altair
IR_df = pd.DataFrame({"Frequency": IR_freqs, "Amplitude": IR_amps})

# Build an Altair line chart for the IR
IR_chart = (
    alt.Chart(IR_df)
    .mark_line(color="#66FCF1")
    .encode(
        x=alt.X(
            "Frequency",
            scale=alt.Scale(
                type="log",      # Logarithmic scale
                domain=[20, 20000]
            )
        ),
        y="Amplitude"
    )
)

# ------------------
# IF USER SUBMITS AUDIO
# ------------------
if audio_value:
    # Reset session state variables when a new audio is uploaded
    if "previous_audio" not in st.session_state or st.session_state["previous_audio"] != audio_value:
        st.session_state["button1"] = False
        st.session_state["button2"] = False
        st.session_state["button3"] = False
        st.session_state["previous_audio"] = audio_value

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    st.write("To efficiently perform convolution, we need to perform Fourier Transform on both your beautiful voice and the IR.")
    
    # Read bytes from the uploaded file
    audio_bytes = audio_value.read()

    # Convert to NumPy array (16-bit PCM is typical)
    # Adjust dtype if your capture widget returns 32-bit or other format
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

    # ------------------
    # FREQUENCY ANALYSIS OF USER AUDIO
    # ------------------
    frequency_domain = np.fft.fft(audio_data)
    frequency_bins = np.fft.fftfreq(len(audio_data), d=1.0 / sample_rate)
    amplitude_spectrum = np.abs(frequency_domain)

    # Filter 20 Hz to 20 kHz
    mask = (frequency_bins >= 20) & (frequency_bins <= 20000)
    freqs = frequency_bins[mask]
    amps = amplitude_spectrum[mask]

    # Create DataFrame for Altair
    df = pd.DataFrame({"Frequency": freqs, "Amplitude": amps})

    # Build Altair chart
    chart = (
        alt.Chart(df)
        .mark_line(color="#66FCF1")
        .encode(
            x=alt.X(
                "Frequency",
                scale=alt.Scale(
                    type="log",
                    domain=[20, 20000]
                )
            ),
            y="Amplitude"
        )
    )

    if st.button("Perform Fourier Transform", use_container_width=True):
        st.session_state["button1"] = not st.session_state["button1"]
        
    if st.session_state["button1"]:

        # Show user audio spectrum
        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
        st.write("This is your beautiful voice in frequency domain.")
        st.altair_chart(chart, use_container_width=True)

        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
        st.write("I have choosen a concert hall reverb for you. Let's see what it looks like in frequency domain.")

        if st.button("Show Impulse Response in Frequency Domain", use_container_width=True):
            st.session_state["button2"] = not st.session_state["button2"]

        if st.session_state["button1"] and st.session_state["button2"]:

            # Show IR spectrum
            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
            st.write("This is the impulse response of a concert hall in frequency domain.")
            st.altair_chart(IR_chart, use_container_width=True)

            st.write("Remember the step above? You multiply two signals in frequency domain!")
            st.latex("Y[e^{j\Omega}] = X[e^{j\Omega}] \cdot H[e^{j\Omega}]")

            if st.button("Multiply the two signals in frequency domain", use_container_width=True):
                st.session_state["button3"] = not st.session_state["button3"]

            if st.session_state["button1"] and st.session_state["button2"] and st.session_state["button3"]:
                # ------------------
                # LINEAR CONVOLUTION IN FREQUENCY DOMAIN
                # ------------------
                # For full linear convolution, length = len(audio) + len(ir) - 1
                N = len(audio_data) + len(IR_data) - 1

                # FFT both signals with the same size N
                audio_fft = np.fft.fft(audio_data, n=N)
                IR_fft = np.fft.fft(IR_data, n=N)

                # Multiply in frequency domain => convolve in time domain
                convolved_spectrum = audio_fft * IR_fft

                # Frequency analysis of the convolved signal
                convolved_amplitude_spectrum = np.abs(convolved_spectrum)
                convolved_frequency_bins = np.fft.fftfreq(N, d=1.0 / sample_rate)

                # Filter 20 Hz to 20 kHz
                convolved_mask = (convolved_frequency_bins >= 20) & (convolved_frequency_bins <= 20000)
                convolved_freqs = convolved_frequency_bins[convolved_mask]
                convolved_amps = convolved_amplitude_spectrum[convolved_mask]

                # Create DataFrame for Altair
                convolved_df = pd.DataFrame({"Frequency": convolved_freqs, "Amplitude": convolved_amps})

                # Build Altair chart for the convolved signal
                convolved_chart = (
                    alt.Chart(convolved_df)
                    .mark_line(color="#66FCF1")
                    .encode(
                        x=alt.X(
                            "Frequency",
                            scale=alt.Scale(
                                type="log",
                                domain=[20, 20000]
                            )
                        ),
                        y="Amplitude"
                    )
                )

                st.altair_chart(convolved_chart, use_container_width=True)

                st.write("And apply the Inverse Fourier Transform!")
                st.latex("Y[e^{j\Omega}] \\longrightarrow y[n]")

                if st.button("CONVOLVE!", use_container_width=True):
                    convolved = np.fft.ifft(convolved_spectrum)

                    # Take the real part, convert to float32 (or int16)
                    convolved_real = np.real(convolved).astype(np.float32)
                    max_val = np.max(np.abs(convolved_real))
                    if max_val > 0:
                        convolved_real = convolved_real / max_val  # Normalize to [-1, 1]
                        convolved_real = (convolved_real * 32767).astype(np.int16)  # Convert to 16-bit

                    # ------------------
                    # APPLY FADE-IN
                    # ------------------
                    fade_duration = int(sample_rate * 1)  # 2 seconds fade-in
                    fade_in = np.linspace(0, 1, fade_duration)
                    convolved_real[:fade_duration] = convolved_real[:fade_duration] * fade_in

                    # ------------------
                    # WRITE TO AN IN-MEMORY WAV FILE
                    # ------------------
                    buffer = io.BytesIO()
                    sf.write(buffer, convolved_real, sample_rate, format='WAV')
                    buffer.seek(0)

                    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
                    st.write("Here's your voice convolved with the impulse response:")
                    st.audio(buffer, format="audio/wav")




st.markdown('<div style="margin-top: 200px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")





