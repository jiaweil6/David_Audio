import streamlit as st
import streamlit.components.v1 as components
from sidebar import sidebar
import io
import soundfile as sf
import librosa
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Phase Vocoder",
    page_icon="📽️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


sidebar()
main_body_logo = "images/icon.png"
st.logo(main_body_logo)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

# Add some padding to prevent content from being hidden behind the navbar
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.markdown("<h2 class='title';'>The algorithm behind trying to finish a 40-min YouTube lecture in 20 mins<br> What is <u>Phase Vocoder</u>?</h2>", unsafe_allow_html=True)

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

subsection = '''Writer: <a href="about" target="_blank" class="author">David Liu</a><br>
Editor: <a href="about" target="_blank" class="author">David Liu</a><br>
Date: 2025-05-14
'''

st.markdown(subsection, unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Motivation")

st.write("""
At some point in our academic career, we all have sped through a lecture video on Youtube at 2x speed cramming all that knowledge the night before the exam. ⚡️
But have you ever wondered how this is possible? 🤔 Going all the way back to how audio is processed in our computer, continous-time audio is sampled at a rate and quantized into discrete values.
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

col_nothing, col_img, col_nothing2 = st.columns([1, 4, 1])   # adjust the width ratio to taste

with col_img:
    st.image("images/sample-waveform.gif", caption="Demo from Google Deepmind", use_container_width="always")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
Let's say we want to speed up the audio by 2x, means we want the length to be half the original. We can simply skip every other sample where the audio is decimated by 2.
Now if we just play the audio using the original sampling rate, we will have the same audio but twice as fast right? Let's visualize it to see if that makes sense!
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

col_nothing, col_img, col_nothing2 = st.columns([1, 4, 1])   # adjust the width ratio to taste

with col_img:
    st.image("images/speed_up_waveform.png", caption="Plot of the waveform before and after speed up by 4x", use_container_width="always")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

with st.expander("Interested in the MATLAB code behind this plot?"):
    st.code("""
clear; close all; clc;

%% Parameters
fs_orig    = 44100;               % original sampling rate
f          = 500;                 % pick your sine freq (500 Hz)
T          = 1/f;                 % one period, in seconds
fs_dec     = 11025;               % decimated rate
skipFactor = fs_orig / fs_dec;    % should be 4

%% Generate exactly one cycle
t_orig = 0 : 1/fs_orig : T;       
x_orig = sin(2*pi*f*t_orig);      

%% Decimate that one-cycle by 4
x_dec = resample(x_orig, fs_dec, fs_orig);
t_dec = (0:numel(x_dec)-1)/fs_dec;

%% 4× Speed-up by skipping
x_fast = x_orig(1:skipFactor:end);
t_fast = (0:numel(x_fast)-1)/fs_orig;

%% Plot
figure('Position',[100 100 800 600]);

subplot(3,1,1);
stem(t_orig, x_orig,'filled');
title('Original @ 44.1 kHz – one cycle');
xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 T]);

subplot(3,1,2);
stem(t_dec, x_dec,'filled');
title('Decimated (↓4) @ 11.025 kHz');
xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 T]);

subplot(3,1,3);
stem(t_fast, x_fast,'filled');
title('4× Speed-Up by Skipping Samples');
xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 T]);

""")
    
st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

audio_value = st.audio_input("Let's try this method out yourself! 🎤")

def resample_audio(audio_value, sample_rate):
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

    return audio_data

if audio_value:
    audio_data = resample_audio(audio_value, 44100)
    audio_data = audio_data[::2]  # decimate by a factor of 2


    st.audio(audio_data, format="audio/wav", sample_rate=44100)
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <p>It is indeed twice as fast! But there is a problem, the pitch is now off. I can't be watching my professor yapping like a chipmunk!
    One way to fix this is to use a specific signal processing algorithm called <u>Phase Vocoder</u>.</p>
    """, unsafe_allow_html=True)
    


st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.subheader("Introduction")

st.markdown("""
<p>In order to understand the <u>Phase Vocoder</u>, we need to first know what a Vocoder is! <u>Vocoder</u> stands for Voice Encoder, it was originally designed to encode and compress
human speech for more efficient transmission. A vocoder typically has two main stages:<br> </p>
""", unsafe_allow_html=True)

st.markdown("""
1. **Analysis (Encoding):**  
   Break speech into parameters describing its frequency and amplitude characteristics. Usually done using Fourier Transform, more specifically the Short-Time Fourier Transform (STFT).

2. **Synthesis (Decoding):**  
   Use these parameters to reconstruct the original or a modified speech signal. This is the fun part! Hang in there!
""")

st.write("""Vocoders can be used for a variety of different applications. This includes Telecommunications, Music, Speech Synthesis (TTS), Speech Modification, and so so much more!""") 

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.markdown("""<p>Now back to the <u>Phase Vocoder</u>! It was introduced by <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6767824" class="author">Flanagan and Golden in 1966</a>.
            How is it special? Frequency resolution is what determines how accurate our pitches are.</p>""", unsafe_allow_html=True)

st.latex(r"""
f_{\text{resolution}} = \frac{f_{\text{sampling rate}}}{N_{\text{window size}}}
""")

st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            
st.write("""
    When using a window size of 1024, and a sampling rate of 44.1 kHz, the frequency resolution is 43 Hz. This is indeed NOT enough to tell apart low-frequency sounds like bass guitar going down to 80Hz.
    Surprisingly, phase vocoders can actually estimate the frequency of a signal to a much higher accuracy using the phase information! And it is also keeping track of all the pitches while speeding up your professor's lecture!
""")


st.write("""
    This is indeed a very confusing topic, as Flanagan and Golden spent 15 pages to explain it in their paper.
    I have written three versions trying to explain it in different ways, hopefully as you start from the easy one, the concept will click for you at the end!
""")

st.divider()

st.subheader("Step by Step Walkthrough")

st.markdown("""
    <p>When processing audio digitally, we almost always use Short-Time Fourier Transform (STFT)  
    to decompose the signal into overlapping windows. Below is a beautiful visualization of the STFT  
    by <a href="https://sethares.engr.wisc.edu/vocoders/phasevocoder.html" class="author">William Sethares</a>.</p>
    """, unsafe_allow_html=True)

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

# Create two columns: wider text on the left, image on the right
col_nothing, col_img, col_nothing2 = st.columns([1, 2.5, 1])   # adjust the width ratio to taste

with col_img:
    st.image(
        "images/dsp_flowchart.jpeg",
        caption="Visualization of the STFT by William Sethares",
        use_container_width=True)      # let Streamlit handle the width

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
    The key essence of the phase vocoder is at the spectral manipulation stage. 
    You signal has been windowed into frames, and then transformed into the frequency domain.
    In case where you have window size of 1024, now you have an array of 513 complex numbers where each number represents the amplitude and phase of a specific frequency from 0 up to the Nyquist frequency.
    Before going straight into equations, maybe take a look at the easy mode to get a general intuition could be a great idea!
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

left, middle, right = st.columns(3)

# Easy mode
if left.button("Easy mode", icon="🍼", use_container_width=True):
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    
    st.write("""1024 samples in the time domains are converted into 513 complex numbers in the frequency domain, each representing a frequency from 0 to 22050 Hz (Nyquist frequency) if we are sampling at 44.1 kHz. 
             Now the array size is still 1024 but anything after index 512 is meaningless. See GOATED MODE for the full explanation!""")

    st.write("""
    Now think of each frequency bin in that array as its own little merry-go-round. 
    Higher frequency bins spins faster, and lower frequency bins spins slower.
    We are going to take two quick snapshots of each merry-go-round, 256 samples apart (hop length of 256 samples), and then deduce how fast it's really spinning. 
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.subheader("Step 1: Two Polaroids", divider=True)

    st.write("""
    Remember that all process below is repeated for each of the 513 frequency bins! 
    """)

    st.markdown("""
<p>1. Photo #1 (previous frame): the red horse is at the north pole <br>
2. Photo #2 (current frame): the red horse is now a bit past east</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p>From these two photos alone we know how many degrees it turned in 256 samples. We will call this <u>actual turn</u>.</p>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.subheader("Step 2: Manufacturer's Speed Sticker", divider=True)

    st.markdown("""
    Each of our merry-go-round has a factory sticker that says it should spin at exactly certain radians per sample.
    If we multiply that sticker speed by 256 samples, we get the <u>expected turn</u> -- where the hourse should have been if the ride was perfectly calibrated. 
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.subheader("Step 3: Calibration Error", divider=True)

    st.markdown("""
    Now compare the photos to the sticker. If the horse went a little farther than predicted, the ride is running hot (sharp), and if it fell short, it's running slow (flat).
    That extra or missing slice of angle is the <u>calibration error</u>.
    """, unsafe_allow_html=True)

    st.markdown("""
    <p>Instead of saying "it overshot by 15 degrees after 256 samples", we'd like to know how many extra degrees per one sample is that?
        So we divide the <u>calibration error</u> by 256 to get the <u>extra degrees per sample</u>.</p>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.subheader("Step 4: Build the Real Speedometer", divider=True)

    st.write("""Finally, the real spin speed of that merry-go-round is simply:""")

    st.latex(r"""
    real\_speed\_per\_sample = sticker\_speed\_per\_sample + extra\_speed\_per\_sample
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <p>Why is finding <u>real speed per sample</u> useful?
    When we later pack slices closer together to play audio twice as fast, we'll ask every merry-go-rouund to simply, 
    rotate by <u> real speed per sample</u> x 128 (hop length / 2) between these two snapshots, not by the sticker amount. </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p>Instead of taking one photo every 256 samples, we just click it quicker to take one every 128 samples.
    Because we know after 256 samples, the horse has rotated to a little past east, let's say 100 degrees, then if the merry-go-round follows the real speed, 
    it should turn 50 degrees after 128 samples! </p>
    """, unsafe_allow_html=True)

    st.write("""
    In this case, because we are using real speed, sharp note stay sharp, and flat notes stay flat!
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.latex(r"""
    extra\_turn = real\_speed\_per\_sample \times new\_hop\_length \space (128)
    """)
    st.latex(r"""
    new\_phase = phase\_previous + extra\_turn
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("""
    Now we simply apply the new phase to every frequency bin in our current frame, then we are good to convert it back to the time domain using ISTFT!
    Remember to change your hop length to half the original to shorten the time duration by half! 
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    



    
# Medium mode
if middle.button("Medium mode", icon="😉", use_container_width=True):
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    st.write("In here you will see a more of a plain-English and pseudo-code style walkthrough of the spectral processing stage of the phase vocoder. Yet with less analogies and descriptions but more technical details.")

    st.write("""
    For every windowed segment with hop length of 256 samples, we run FFT to get the frequency domain representation.
    Here for each frequency bin, we have a complex number representing the amplitude and phase of that frequency.
    Again, our goal here is to find the real speed per sample for each frequency bin so we can apply it to any hop length to time compress or stretch the audio with the correct pitch.
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown(
    r"""
    $$
    raw\_phase\_change =
    phase\_now -
    phase\_previous\;
    (\text{wraps into } -\pi \dots \pi)
    $$
    """,
    unsafe_allow_html=True,
    )

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    In cases where we are sampling at 44.1 kHz, and our window size is 1024, we will have 513 frequency bins.
    Where each bin represents a frequency, and that we will call it <u>theoretical_bin_pitch</u>.
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.latex(r"""
    theoretical\_bin\_pitch[m] = \frac{f_{sampling}}{N_{window}} \times m
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    # Parameters
    fs = 44_100   # sampling rate (Hz)
    N  = 1024     # FFT length

    # Choose 5 evenly spaced bin indices from 0 to N/2 (inclusive)
    bin_indices = np.linspace(0, N//2, num=5, dtype=int)

    # Create DataFrame
    df_sample = pd.DataFrame({
        "Bin Index": bin_indices,
        "Frequency (Hz)": bin_indices * fs / N
    })

    col_nothing, col_img, col_nothing2 = st.columns([1, 2.5, 1])   # adjust the width ratio to taste

    with col_img: 
        st.dataframe(df_sample)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    Now we need to find the <u>expected phase change</u> for each frequency bin.
    We know the speed (<u>theoretical_bin_pitch</u>) of each bin, and the time it takes to travel (<u>hop_length</u>).
    Simply multiply the two to get the distance, the <u>expected phase change</u> after your current frame.
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.latex(r"""
    expected\_phase\_change = theoretical\_bin\_pitch \times hop\_length
    """)

    st.latex(r"""
    extra\_phase\_change = raw\_phase\_change - expected\_phase\_change
    """)

    st.latex(r"""
    real\_phase\_change\_per\_sample = theoretical\_bin\_pitch + \frac{extra\_phase\_change}{hop\_length}
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    Now, we have everything we want from this current frame. It's time to speed it up! 
    Because we have chopped a piece of audio into overlapping frames, we place them back together in the similar overlapping fashion.
    If we overlap it by the same amount, nothing is changed, we get the original audio back.
    But our goal here is to speed it up, so we overlap it by less the original amount.
    We can define our <u>new hop length</u> as:
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)

    st.latex(r"""
    new\_hop\_length = \frac{hop\_length}{r}
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("""
    Where r is the speed factor, greater than 1 means we are speeding up, less than 1 means we are slowing down.
    More STFT and overlapping methods can be found in the next section!
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.latex(r"""
    new\_phase = phase\_previous + real\_phase\_change\_per\_sample \times new\_hop\_length
    """)

    st.latex(r"""
    complex\_value = original\_magnitude \times e^{j \cdot new\_phase}
    """)
    
    st.latex(r"""
    frame\_in\_time = inverse\_fft(complex\_value)
    """)

    st.latex(r"""
    windowed\_frame = frame\_in\_time \times window\_function
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("""Due to the nature of windows, they are designed so overlapping pieces adds up to "one". 
             Glueing all the frames back together, is simply adding them together. """)
    
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    
    st.code("""
    for each new_frame_index:
        output[start_sample : start_sample + window_length] += windowed_frame
        start_sample += new_hop_length
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    



    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)






# GOATED MODE
if right.button("GOATED MODE", icon="🔥", use_container_width=True):
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("""
    Below is a step-by-step walkthrough of the clasic phase-vocoder time-scale-modification (TSM) algorithm.
    Here are some of the parameters we will be using:
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    col_nothing, col_img, col_nothing2 = st.columns([0.5, 6, 0.5])   # adjust the width ratio to taste

    with col_img: 
        st.markdown(r"""
        | Symbol | Meaning | Typical choice |
        |--------|---------|----------------|
        | $x[n]$ | original discrete-time signal (length $N$) | any audio |
        | $r$ | speed-up factor ($>1$) | 1.5 – 4 |
        | $N_{\text{FFT}}$ | DFT length (window length) | 1024 – 4096 |
        | $H_a$ | analysis hop (original hop length) | $N_{\text{FFT}} / 4$ (Hann) |
        | $H_s$ | synthesis hop | $H_a / r$ |
        | $w[n]$ | analysis / synthesis window | Hann; satisfies COLA |
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("""
    COLA here stands for Constant-Overlap-Add where:
    """)

    st.latex(r"""
    \sum_k = w[n - kH_a] = const.
    """)

    col_nothing, col_img, col_nothing2 = st.columns([1, 4, 1])   # adjust the width ratio to taste

    with col_img: 
        st.image("images/ola_animation.gif", use_container_width=True)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    with st.expander("Interested in the Python code behind this gif?"):
        st.code("""
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.signal.windows import kaiser
            from matplotlib.animation import FuncAnimation

            # Parameters
            M = 33                # window length
            beta = 8              # Kaiser beta parameter
            w = kaiser(M, beta)   # 33-point Kaiser window
            R = 6                 # hop size
            N = 99               # total signal length

            # Number of frames that fit into length N
            numFrames = (N - M) // R + 1

            # Precompute window positions
            positions = [k * R for k in range(numFrames)]

            # Precompute the full overlap-added sum to determine y-axis limit
            full_ola = np.zeros(N)
            for pos in positions:
                full_ola[pos:pos+M] += w
            max_val = np.max(full_ola)

            # Prepare figure and axes
            plt.rcParams['animation.html'] = 'jshtml'
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_xlim(0, N)
            ax.set_ylim(0, max_val * 1.1)
            ax.set_title('Overlap-Add')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')

            # Update function for each frame
            def update(k):
                ax.clear()
                ax.set_xlim(0, N)
                ax.set_ylim(0, max_val * 1.1)
                ax.set_title('Overlap-Add')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Amplitude')
                
                # Plot previous windows as dotted red
                for i in range(k):
                    pos_i = positions[i]
                    idx_i = np.arange(pos_i, pos_i + M)
                    ax.plot(idx_i, w, 'r--')
                
                # Plot current window as solid red
                pos_k = positions[k]
                idx_k = np.arange(pos_k, pos_k + M)
                ax.plot(idx_k, w, 'r-', linewidth=1.5)
                
                # Compute and plot cumulative sum up to k
                cumulative = np.zeros(N)
                for i in range(k+1):
                    pos_i = positions[i]
                    cumulative[pos_i:pos_i+M] += w
                ax.plot(np.arange(N), cumulative, 'b-', linewidth=2)
                
            
            # Create the animation
            anim = FuncAnimation(fig, update, frames=numFrames, interval=200, blit=False)

            # Display the animation
            anim 
        """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.subheader("1. Analysis STFT", divider="grey")

    st.markdown("""
    <p>1. Frame the signal:</p>
    """, unsafe_allow_html=True)

    st.latex(r"""
    x_k[n] = w[n] x[n + kH_a], \space n=0,...,N_{FFT} - 1, \space k=0,...,K-1
    """)

    st.latex(r""" K \simeq \lceil N / H_a \rceil""")

    st.markdown("""
    <p>2. Obtain complex spectra:</p>
    """, unsafe_allow_html=True)

    st.latex(r"""
    X_k[m] = FFT(x_k[n]), \space m=0,...,N_{FFT} - 1
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("We separate it into magnitude and phase:")

    st.latex(r""" 
    A_k[m] = |X_k[m]|
    """)

    st.latex(r"""
    \phi_k[m] = \angle X_k[m]
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("""
    For implementation and efficiency purposes, we usually use the real FFT (rFFT) which outputs half the number of bins,
    because of FFT symmetry. When the time-domain signal x[n] is real, the output of discrete Fourier transform exhibites Hermitian (complex conjugate) symmetry, where:
    """)

    st.latex(r"""
    X[m] = \overline{X\!\left[N_{\mathrm{FFT}}-m\right]}, \space 0 < m < N_{FFT}
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.code("""
# Full complex FFT (N outputs)
X_full = np.fft.fft(frame, n=N_FFT)

# Real FFT (N/2+1 outputs, unique bins only)
X_r = np.fft.rfft(frame, n=N_FFT)   # indices 0 … N_FFT//2""")
    
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("""
    So technically, we only need the first half of the complex FFT output which saves us some computation! 
    And the mathamatical proof is as follows:
    """)

    with st.expander("I know you love math, here's the proof!"):
        col_1, col_2 = st.columns([1, 1])

        with col_1:
            st.latex(r"""
            X[N - m] = \sum_{n=0}^{N-1} x[n]e^{-j \frac{2\pi}{N}(N-m)n}
            """)
            st.latex(r"""
            X[N - m] = \sum_{n=0}^{N-1} x[n]e^{-j2\pi n}e^{j\frac{2\pi}{N}mn}
            """)
            st.latex(r"""
            X[N - m] = \sum_{n=0}^{N-1} x[n]e^{j\frac{2\pi}{N}mn}
            """)
            
        with col_2:
            st.latex(r"""
            \overline{X[m]} = \overline{\sum_{n=0}^{N-1} x[n]e^{-j \frac{2\pi}{N}mn}}
            """)
            st.latex(r"""
            \overline{X[m]} = \sum_{n=0}^{N-1} \overline{x[n]} \overline{e^{-j \frac{2\pi}{N}mn}}
            """)
            st.latex(r"""
            \overline{X[m]} = \sum_{n=0}^{N-1} x[n] e^{j \frac{2\pi}{N}mn}
            """)

        st.latex(r"""
        X[N - m] = \overline{X[m]}
        """)

        st.latex(r"""
        X[m] = \overline{X[N - m]}
        """)

        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
        
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)




    

    st.subheader("2. Track true instantaneous frequency per bin", divider="grey")

    st.write("""
    The FFT's raw phases are discontinuous across frames. We unwrap them and measure the actual bin-wise frequency:
    """)

    st.markdown("""
    <p>1. Compute phase advance that happened between consecutive frames:</p>""", unsafe_allow_html=True)

    st.latex(r"""\Delta \phi_k[m] = [\phi_k[m]-\phi_{k-1}[m]]_\pi""")
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <p>2. Subtract the expected advance:</p>""", unsafe_allow_html=True)

    st.latex(r"""\omega_m = 2\pi m / N_{FFT} \longrightarrow \delta_k[m] = \Delta \phi_k[m] - \omega_m H_a""")
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <p>3. Instantaneous frequency estimate:</p>""", unsafe_allow_html=True)

    st.latex(r"""
    \hat{\omega}_m = \omega_m + \frac{\delta_k[m]}{H_a}
    """)
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.write("""This information is what lets us time-scale while keeping pitch fixed.""")

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)





    st.subheader("3. Phase accumulation", divider="grey")

    st.markdown("""
    <p>1. Accumulate phase using the instantaneous frequency from the original analysis frame that contributed:""", unsafe_allow_html=True)

    st.latex(r"""\Phi_k[m] = \Phi_{k - 1}[m] + \hat{\omega}_k [m] H_s""")

    st.latex(r"""\Phi_0[m] = \phi_0[m] """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <p>2. Recombine magnitude with new phase:""", unsafe_allow_html=True)

    st.latex(r"""
    Y_k[m] = A_k[m] e^{j \Phi_k[m]}
    """)
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)






    st.subheader("4. Inverse and overlap-add", divider="grey")
    
    st.markdown("""
    <p>1. Inverse FFT:</p>""", unsafe_allow_html=True)

    st.latex(r"""
    y_k[n] = iFFT(Y_k[m])
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <p>2. Window with the same w[n] used in analysis:</p>""", unsafe_allow_html=True)

    st.markdown("""<p>3. Overlap-add:</p>""", unsafe_allow_html=True)

    st.latex(r"""
    y[n] \mathrel{+}= w[n] y_k[n - kH_s]
    """)

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    st.markdown("""
    Because $\omega[n]$ satisfies the COLA property and we respected consistent phases, the overlaps sum smoothly, yielding an output signal whose duration is 
    $N/r$ but whose spectral envelopes (hence perceived pitch/formants) remain intact!""")

    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    
    
    
    
    
    
    
    
    
    







st.divider()
st.subheader("Intuition Behind Changing the Hop Length")


st.markdown('<div style="margin-top: 200px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")
