import streamlit as st
from sidebar import sidebar
import numpy as np
import pandas as pd
import altair as alt
import io
import soundfile as sf
import librosa

st.set_page_config(
    page_title="Convolution",
    page_icon="🎤",
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
main_body_logo = "images/icon.png"
st.logo(main_body_logo)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
# Add some padding to prevent content from being hidden behind the navbar
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.markdown("<h2 class='title';'>Algorithm Behind Realistic Reverb? <br> What is <u>Convolution</u>?</h2>", unsafe_allow_html=True)

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

subsection = '''Writer: <a href="about" target="_blank" class="author">David Liu</a><br>
Editor: <a href="about" target="_blank" class="author">David Liu</a><br>
Date: 2025-01-03
'''

st.markdown(subsection, unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Introduction")

st.write("""
Ever wonder how your computer magically transports your voice into a grand concert hall? 🪄
Is it just stretching out the tail end of your vocals to create that luscious reverb? 
Or how your amp simulator seems to capture the exact sound of that ridiculously expensive amp you’ve been eyeing? 🤑
The secret behind the scenes is something called convolution—and it’s a total game-changer. 🔥
""")

st.markdown("""
<p><u>Convolution</u> is a special mathematical operation that combines two signals to produce a third signal. 
You’ve probably heard the term thrown around in convolution reverb or IR (impulse response) simulation. 
No one wants to slog through the entire Wikipedia page just to end up more confused, right? 😅</p>
""", unsafe_allow_html=True)

st.write("""
Convolution of two signal $x$ and $h$ is written $x \\ast h$ where:
""")

st.latex(r"(x \ast h)(t) = \int_{-\infty}^{\infty} x(\tau) h(t - \tau) d\tau")

st.write("""
Don’t bail on me just yet—I know you’re probably scratching your head over that funky-looking integral. 🤮 Trust me, I was just as confused the first time I saw it!
Think of $x$ as your input (or “dry”) signal and $h$ as a mystery box or effect that the signal passes through. That’s all there is to it for now—no need to freak out! 👍
""")

st.image("images/flow-chart.png", use_container_width="always")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Intuition of Impulse Response")

st.write("""
    As I mentioned, $h$ is basically the effect we want to apply to the dry signal. 
    Sometimes we refer to it as the “system” or the “impulse response.” 
    Mathematically, the impulse response is the output you get when you feed an impulse signal into the system.
""")

st.latex("x(t) = \delta(t)")
st.latex("(x \\ast h) (t) = y(t) = h(t)")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
    Imagine an impulse signal as a single spike of sound—maybe a single drum hit 🥁 or a balloon pop 🎈, which is about as close as you can get to a Dirac delta function in real life.
    Now picture playing that spike in a concert hall: the sound bounces off the walls and ceiling, so that short spike—plus the concert hall’s reverberation—becomes the impulse response. 
    While this isn’t exactly how you’d measure an impulse response in real life, it’s still a great way to wrap your head around the math behind it.
""")

st.write("""
    Now that we’ve captured the impulse response (or the “effect”), we can apply it to our dry signal.
    To get that epic concert hall reverb in our final output, we simply convolve the dry signal with the impulse response. And voilà—instant big-stage vibes! 😌
""")

st.latex("y(t) = x(t) \\ast h(t)")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Practical Scenarios")

st.write("""
    In the digital world 💻, we can’t directly convolve continuous signals—our computers have only so much horsepower! 
    Instead, we break (discretize) the signal into samples and perform a discrete convolution. 
    That funky integral you saw earlier transforms into a summation in the discrete domain.
""")

st.latex("y[n] = \\sum_{k=0}^{N-1} x[k] h[n-k]")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
    Don't calculate your $y[n]$ just yet, imagine your $N$ is a billion and you have a super long signal. 
    That summation would take forever! How can this be done in real-time?
    We engineers have a trick to work around this 😉. All signals have two domains: the time domain and the frequency domain.
""")   

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.image("images/domain.jpeg", caption="Time Domain vs. Frequency Domain image from Keysight", use_container_width="always")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.markdown("""
    <p>Convolution in the time domain is actually multiplication in the frequency domain! 🤯 Much simpler now, but how do we find the signal in its frequency domain?</p>
""", unsafe_allow_html=True)

st.latex("x[n] \\ast h[n] \longleftrightarrow X[f] \cdot H[f]")

st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

with st.expander("Interested in the mathematical proof behind this?"):
    st.latex("y[n] = x[n] \\ast h[n] = \sum_{k=-\infty}^{\infty}x[k]h[n-k]")
    st.latex("Y(e^{jw}) = \sum_{n=-\infty}^{\infty}y[n]e^{-jwn}=\sum_{n=-\infty}^{\infty}\left(\sum_{k=-\infty}^{\infty}x[k]h[n-k]\\right)e^{-jwn}")
    st.latex("= \sum_{k=-\infty}^{\infty}x[k]\sum_{n=-\infty}^{\infty}h[n-k]e^{-jwn}")
    st.latex("= \sum_{k=-\infty}^{\infty}x[k]\sum_{n=-\infty}^{\infty}h[m]e^{-jw(m+k)}")
    st.latex("= \sum_{k=-\infty}^{\infty}x[k]\sum_{m=-\infty}^{\infty}h[m]e^{-jwm}e^{-jwk}")
    st.latex("=\sum_{k=-\infty}^{\infty}x[k]e^{-jwk}\sum_{m=-\infty}^{\infty}h[m]e^{-jwm}")
    st.latex("=X(e^{jw})\cdot H(e^{jw})")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.markdown("""<p>
    This is rather technical but very, very efficient.
    <a href="https://www.jezzamon.com/fourier/" class="author">Fourier Transform</a>, more specifically FFT (Fast Fourier Transform), is sometimes called the most important algorithm of all time.
    I’m not going to scare you with big equations here, but if you’re interested, check out the fantastic demo by Jez Swanson in the link above. 🫡
    It essentially converts a signal from its time domain to its frequency domain. Now, instead of time/sample on the x-axis, we have frequency (in Hz) on the x-axis.
</p>""", unsafe_allow_html=True)

st.latex("x[n] \\longrightarrow X[f]")
st.latex("h[n] \\longrightarrow H[f]")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)


st.write("Vice versa—after multiplying two signals in the frequency domain, we perform the Inverse Fourier Transform to get our time-domain signal back, which we can then blast through a loudspeaker! 🔈")

st.latex("Y[f] = X[f] \cdot H[f]")
st.latex("Y[f] \\longrightarrow y[n]")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
In order for your DAW to process the entire convolution in real time, 
it chops your signal into chunks and performs the same process shown above on each chunk at a speed so fast you can barely notice it. ⚡️
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Ready to try it out yourself?")
st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

# ------------------
# AUDIO INPUT
# ------------------
audio_value = st.audio_input("Sing your heart out here 🎤")

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
        y=alt.Y(
            "Amplitude",
            axis=alt.Axis(labels=False, ticks=False)  # Hide labels and ticks
        )
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
        st.session_state["button4"] = False
        st.session_state["button5"] = False
        st.session_state["previous_audio"] = audio_value

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    st.write("1️⃣ Remember how to efficiently perform convolution? We need to perform a Fourier Transform on both your beautiful voice and the IR.")
    
    # 1) Read the raw bytes from your uploaded file
    audio_bytes = audio_value.read()

    # 2) Create an in-memory buffer from those bytes
    buffer = io.BytesIO(audio_bytes)

    # 3) Use soundfile to read the data and sample rate
    audio_data, audio_sample_rate = sf.read(buffer)
    # audio_data is now a NumPy float array (float32 or float64),
    # and audio_sample_rate is whatever was specified in the file.

    # 4) If the file sample rate != 44.1 kHz, resample using librosa
    if audio_sample_rate != sample_rate:
        audio_data = librosa.resample(audio_data, 
                                    orig_sr=audio_sample_rate, 
                                    target_sr=sample_rate)
        audio_sample_rate = sample_rate

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
    amps /= np.max(amps)

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
            y=alt.Y(
                "Amplitude",
                axis=alt.Axis(labels=False, ticks=False)  # Hide labels and ticks
            )
        )
    )

    if st.button("Perform Fourier Transform", use_container_width=True):
        st.session_state["button1"] = not st.session_state["button1"]
        
    if st.session_state["button1"]:
        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
        # Show user audio spectrum
        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
        st.write("This is your voice in frequency domain.")
        st.altair_chart(chart, use_container_width=True)
        st.write("I’ve chosen a concert hall reverb for you. Let’s see what it looks like in the frequency domain!")

        if st.button("Show Impulse Response in Frequency Domain", use_container_width=True):
            st.session_state["button2"] = not st.session_state["button2"]

        if st.session_state["button1"] and st.session_state["button2"]:

            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
            # Show IR spectrum
            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
            st.write("This is the impulse response of a concert hall in the frequency domain.")
            st.altair_chart(IR_chart, use_container_width=True)

            st.write("2️⃣ Remember from earlier? You multiply two signals in the frequency domain!")
            st.latex("Y[f] = X[f] \cdot H[f]")
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

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
                convolved_amps /= np.max(convolved_amps)

                # Create DataFrame for Altair
                convolved_df = pd.DataFrame({"Frequency": convolved_freqs, "Amplitude": convolved_amps})

                # Build Altair chart for the convolved signal
                convolved_chart = (
                    alt.Chart(convolved_df)
                    .mark_line(color="#FF5733")
                    .encode(
                        x=alt.X(
                            "Frequency",
                            scale=alt.Scale(
                                type="log",
                                domain=[20, 20000]
                            )
                        ),
                        y=alt.Y(
                            "Amplitude",
                            axis=alt.Axis(labels=False, ticks=False)  # Hide labels and ticks
                        )
                    )
                )

                # Overlay the convolved chart on top of the original chart
                combined_chart = chart + convolved_chart
                
                st.markdown('<div style="margin-bottom: 80px;"></div>', unsafe_allow_html=True)
                st.altair_chart(combined_chart, use_container_width=True)

                st.markdown('''<p>3️⃣ Now we have both the <span style="color: #66FCF1;">dry signal</span> and the <span style="color: #FF5733;">wet signal</span> in the frequency domain. 
                            It’s time to convert them back into the time domain (and make them playable) by performing the Inverse Fourier Transform!</p>''', unsafe_allow_html=True)

                st.latex("Y[f] \\longrightarrow y[n]")
                st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

                if st.button("Inverse Fourier Transform!", use_container_width=True):
                    st.session_state["button4"] = not st.session_state["button4"]

                if st.session_state["button1"] and st.session_state["button2"] and st.session_state["button3"] and st.session_state["button4"]:
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

                    st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
                    st.write("Here's your voice convolved with the impulse response:")
                    st.audio(buffer, format="audio/wav")
                    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
                    st.write("This is the fully wet signal, but you might not need this much reverb. Pick how much of that concert hall vibe you want and dial it in!")
                    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
                    reverb_amount = st.slider("0 for fully dry, 100 for fully wet", min_value=0, max_value=100, value=50, step=5)
                    st.latex("output = (1 - \\alpha) \cdot x(t) + \\alpha  \cdot y(t)")
                    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
                    st.markdown('''<p><span style="color: #66FCF1;">x(t) is the dry signal</span>, <span style="color: #FF5733;">y(t) is the wet signal</span>, and 
                                alpha is the amount of reverb you want to apply, scaled from 0 to 1.</p>''', unsafe_allow_html=True)
                    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

                    if st.button("Mix up the wet and the dry to find your perfect balance!", use_container_width=True):
                        st.session_state["button5"] = not st.session_state["button5"]

                    if st.session_state["button5"]:
                        # 1) Ensure both signals are float32 to avoid weird type mismatches
                        dry = audio_data.astype(np.float32)
                        wet = convolved_real.astype(np.float32)

                        # 2) Pad the shorter one so both have the same length (keep the full reverb tail)
                        max_length = max(len(dry), len(wet))
                        dry = np.pad(dry, (0, max_length - len(dry)), 'constant')
                        wet = np.pad(wet, (0, max_length - len(wet)), 'constant')

                        # 3) Mix them
                        alpha = reverb_amount / 100.0  # from 0.0 to 1.0
                        mixed = (1.0 - alpha) * dry + alpha * wet

                        # 4) Normalize to prevent clipping or extremely low volume
                        mixed_normalized = normalize_gain(mixed, target_peak=0.9)

                        # 5) (Optional) Convert to 16-bit integer if you need WAV playback
                        final_signal = (mixed_normalized * 32767).astype(np.int16)

                        
                        st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
                        st.audio(final_signal, format="audio/wav", sample_rate=sample_rate)
                        st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                        st.write('''
                            All of this might seem like magic! Now you can sing from home and sound as if you’re in a concert hall.
                            
                            We often take for granted the amount of work that goes into the technology behind making music. But sometimes, it’s fascinating to peek under the hood of the digital world and see how things really work.
                            
                            I hope this blog post helps you understand the convolution process in audio and how it runs behind the scenes on your computer. If you have any questions, feel free to reach out!
                        ''')

                        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

                        with st.expander("Interested in the code that runs behind the scenes when you press those buttons?"):
                            st.code("""
                                # 1) Convert raw audio bytes to NumPy int16
                                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

                                # 2) Determine the length needed for full convolution
                                N = len(audio_data) + len(ir_data) - 1

                                # 3) FFT both signals (zero-padded to size N)
                                audio_fft = np.fft.fft(audio_data, n=N)
                                ir_fft = np.fft.fft(ir_data, n=N)

                                # 4) Multiply in frequency domain (convolution in time domain)
                                convolved_spectrum = audio_fft * ir_fft

                                # 5) Inverse FFT and take real part
                                convolved_time = np.fft.ifft(convolved_spectrum)
                                convolved_real = np.real(convolved_time).astype(np.float32)

                                # 6) Normalize to int16 range
                                peak = np.max(np.abs(convolved_real))
                                if peak > 0:
                                    convolved_real /= peak  # scale to [-1, 1]
                                convolved_int16 = (convolved_real * 32767).astype(np.int16)

                                # 7) Write to an in-memory WAV file
                                buffer = io.BytesIO()
                                sf.write(buffer, convolved_int16, sample_rate, format='WAV')
                                buffer.seek(0)
                            """)




st.markdown('<div style="margin-top: 200px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")





