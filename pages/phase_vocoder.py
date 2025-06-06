import streamlit as st
from sidebar import sidebar
import io
from io import BytesIO
import soundfile as sf
import librosa
import altair as alt
import numpy as np
import pandas as pd
from scipy.signal import stft, istft, get_window, resample
from scipy.io.wavfile import write as wav_write
from scipy.io import wavfile

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
Date: 2025-06-06
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

""", language="matlab")
    
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
    <p>It is indeed twice as fast! But there is a problem, the pitch is now off. I can't be watching my professor yapping like a chipmunk! 🐿️
    One way to fix this is to use a specific signal processing algorithm called <u>Phase Vocoder</u>.</p>
    """, unsafe_allow_html=True)
    


st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

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
    This is indeed a very confusing topic, as Flanagan and Golden spent 15 pages to explain it in their paper 🤯.
    I have written three versions trying to explain it in different ways, hopefully as you start from the easy one, the concept will click for you at the end ☺️!
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

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
    Before going straight into equations, maybe take a look at the easy mode to get a general intuition could be a great idea 💡!
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

left, middle, right = st.columns(3)

# Easy mode
if left.button("Easy mode", icon="🍼", use_container_width=True):
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    
    st.write("""1024 samples in the time domains are converted into 513 complex numbers in the frequency domain, each representing a frequency from 0 to 22050 Hz (Nyquist frequency) if we are sampling at 44.1 kHz. 
             Now the array size is still 1024 but anything after index 512 is meaningless. See GOATED MODE 🔥 for the full explanation!""")

    st.write("""
    Now think of each frequency bin in that array as its own little merry-go-round 🎠. 
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
    it should turn 50 degrees after 128 samples! 🤯</p>
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
    
    
    
    
    
    
    
    
    
    





st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
st.subheader("Intuition Behind Changing the Hop Length")

st.write("""
Changing the hop length to speed up and down was very confusing for me at first.
It's tricky to understand why no information is lost in the frequency domain.
The animation below should help you understand why. 
We don't really care if we lose a few cycles of the same frequency as long as the wavelength remains the same.
Instead of signal cycling through 2 periods, the speed-up version cycles through 1 or less to achieve this. 
Where as in the naive implementation, where we just decimate the signal, all wavelengths are scaled as well.
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

col_nothing, col_img, col_nothing2 = st.columns([1, 4, 1])   # adjust the width ratio to taste

with col_img: 
    st.image("images/ola_demo.gif", use_container_width=True, caption="Aperiodic chirp signal with different synthesis hop lengths")

with st.expander("A significant amount of work went into making this animation! 😭"):
    st.code("""
    !pip install --upgrade --quiet numpy
    !pip install --upgrade --quiet matplotlib         # plotting + animation
    !pip install --upgrade --quiet scipy              # signal processing (stft, get_window)
    !pip install --upgrade --quiet ipython            # IPython.display for HTML()
    !pip install --upgrade --quiet pillow             # GIF/PNG writer for anim.save()
    """)
    st.code("""
import matplotlib as mpl
mpl.rcParams['animation.html'] = 'jshtml'
from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
from matplotlib.animation import FuncAnimation

# ---------------- 1. Source signal (aperiodic chirp) --------------------------
from scipy.signal import chirp
from scipy.signal import windows          # <- has Tukey

fs       = 44_100
duration = 0.05
t        = np.linspace(0, duration, int(fs*duration), endpoint=False)

x = chirp(t, f0=10, f1=1000, t1=duration, method='linear')
x *= windows.tukey(len(t), alpha=0.3)      # smooth edges

# ---------------- 2. STFT settings -------------------------------------------
n_fft   = 512
hop_a   = 128                       # analysis hop
window  = get_window('hann', n_fft)
_, _, Zxx = stft(x, fs, window=window, nperseg=n_fft,
                 noverlap=n_fft-hop_a, boundary=None)
n_bins, n_frames = Zxx.shape

# ---------------- 3. Build PV frames for each rate ---------------------------
def pv_frames(rate):
    time_steps = np.arange(0, n_frames, rate)      # decimate reading path
    omega      = 2 * np.pi * hop_a * np.arange(n_bins) / n_fft
    phase_acc  = np.angle(Zxx[:, 0]).copy()
    frames     = []
    for ts in time_steps:
        t0, t1 = int(np.floor(ts)), min(int(np.floor(ts))+1, n_frames-1)
        dt     = ts - t0
        mag    = (1-dt)*np.abs(Zxx[:, t0]) + dt*np.abs(Zxx[:, t1])
        pd     = np.angle(Zxx[:, t1]) - np.angle(Zxx[:, t0]) - omega
        pd    -= 2*np.pi * np.round(pd / (2*np.pi))
        phase_acc += omega + pd
        Y = mag * np.exp(1j*phase_acc)
        frames.append(np.fft.irfft(Y, n_fft) * window)
    return frames

rates = [1.0, 2.0, 4.0]             # 1×, 2×, 4× speeds
frames_list = [pv_frames(r) for r in rates]
hops_s      = [hop_a] * len(rates)  # synthesis hop constant
n_frames_list = [len(fr) for fr in frames_list]

# ---------------- 4. Pre-compute full outputs -------------------------------
out_full = []
for frames, hop_s in zip(frames_list, hops_s):
    y_len = len(frames) * hop_s + n_fft
    buf = np.zeros(y_len)
    for idx, fr in enumerate(frames):
        buf[idx*hop_s: idx*hop_s + n_fft] += fr
    out_full.append(buf)

global_max = max(np.max(np.abs(b)) for b in out_full)

# ---------------- 5. Figure --------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(9, 12))
plt.tight_layout()
ax_sig, ax1x, ax2x, ax4x = axes

# Row 1: original + analysis window
ax_sig.plot(t, x, color='black')
ax_sig.set_xlim(0, duration)
ax_sig.set_ylabel('Original')
ax_sig.set_title('Original', fontsize=12, weight='bold')
win_patch, = ax_sig.plot([], [], color='red', lw=2)

# OA axes configuration
oa_axes   = [ax1x, ax2x, ax4x]
labels    = ['1× OA', '2× OA', '4× OA']
titles    = ['1× Speed', '2× Speed', '4× Speed']
buf_lines, frame_lines = [], []

# Determine common x-limit: original length
orig_len_sec = len(out_full[0]) / fs
for ax, lbl, ttl in zip(oa_axes, labels, titles):
    ax.set_xlim(0, orig_len_sec)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel(lbl)
    ax.set_title(ttl, fontsize=12, weight='bold')
    bl, = ax.plot([], [], lw=2)
    fl, = ax.plot([], [], lw=2, alpha=0.65)
    buf_lines.append(bl)
    frame_lines.append(fl)

ax4x.set_xlabel('Time (s)')

# Buffers reused
buf_work = [np.zeros_like(b) for b in out_full]

def init():
    win_patch.set_data([], [])
    for bl, fl in zip(buf_lines, frame_lines):
        bl.set_data([], [])
        fl.set_data([], [])
    return (win_patch, *buf_lines, *frame_lines)

def update(i):
    # ---- slide window on original signal -----
    start = min(i*hop_a, len(x)-n_fft)
    win_patch.set_data(t[start:start+n_fft], x[start:start+n_fft])

    # ---- update each OA plot -----------------
    for idx, rate in enumerate(rates):
        hop_s   = hops_s[idx]
        frames  = frames_list[idx]
        n_fr    = n_frames_list[idx]
        y_len   = len(out_full[idx])
        buf     = buf_work[idx]; buf[:] = 0

        j = min(int(i / rate), n_fr-1)
        # overlap-add up to frame j
        for k in range(j):
            pos = k * hop_s
            buf[pos:pos+n_fft] += frames[k]

        # normalise
        buf_n   = buf / global_max
        frame_n = frames[j] / global_max

        buf_x   = np.arange(y_len) / fs
        frame_x = (np.arange(n_fft) + j*hop_s) / fs

        buf_lines[idx].set_data(buf_x, buf_n)
        frame_lines[idx].set_data(frame_x, frame_n)

    return (win_patch, *buf_lines, *frame_lines)

# --- just before you create the animation ------------------------------------
base_frames   = list(range(n_frames))      # 0 … n_frames-1
hold_ms       = 2000                       # how long to hold (milliseconds)
interval_ms   = 100                        # your existing interval
hold_extra    = hold_ms // interval_ms     # number of extra frames

frame_seq = base_frames + [n_frames-1] * hold_extra
# ------------------------------------------------------------------------------

anim = FuncAnimation(
        fig, update,
        frames=frame_seq,      # use the custom sequence
        init_func=init,
        blit=True,
        interval=interval_ms,
        repeat=True            # loop after the hold segment
)

HTML(anim.to_jshtml())

# Display *inside* the notebook
display(HTML(anim.to_jshtml()))

    """)


st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
As you can see, some wiggles in the time-domain amplitude are introduced.
That is completely normal in the text-book phase vocoder implementation.
This could be from many different sources, including window shape, COLA sum, frame decimation, etc.
Perfect loudness preservation requires extra gain-correction or hybrid approaches, which many modern PV libraries (e.g. RubberBand, librosa’s phase_vocoder) add on top of the classic algorithm.
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.subheader("Try it out yourself!")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

def phase_vocoder(x, speed: float, fs: int = 44_100,
                  n_fft: int = 2048, hop: int | None = None) -> bytes:
    """
    Time-stretch (or compress) a signal with a phase-vocoder.

    Parameters
    ----------
    x      : np.ndarray
        1-D (mono) or 2-D (channels × samples) audio, float32/float64 in [-1, 1].
    speed  : float
        Playback speed (>1 = faster/shorter, <1 = slower/longer).
    fs     : int, default 44100
        Sample rate of *x*.
    n_fft  : int, default 2048
        STFT window length.
    hop    : int | None, default n_fft//4
        Analysis hop.  If None, uses *n_fft // 4*.

    Returns
    -------
    bytes
        16-bit WAV container holding the processed audio.
    """
    if hop is None:
        hop = n_fft // 4

    # ------ mono-ise & make float32 ------
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0)

    # ------ analysis ------
    win = get_window("hann", n_fft, fftbins=True)
    _, _, Z = stft(x, fs=fs, window=win, nperseg=n_fft,
                   noverlap=n_fft - hop, boundary=None, padded=False)

    # How much we *stretch* the sound on the time axis.
    # speed 2.0 (twice as fast)  -> stretch 0.5 (half the frames)
    stretch = 1.0 / speed

    # Target number of frames in the output spectrogram
    n_frames_out = int(np.ceil(Z.shape[1] * stretch))

    # Phase-accumulator & expected phase advance between frames
    phi = np.angle(Z[:, 0])
    dphi = np.angle(Z[:, 1:]) - np.angle(Z[:, :-1])
    dphi = np.unwrap(dphi, axis=1)

    # Container for the time-scaled STFT
    Z_out = np.empty((Z.shape[0], n_frames_out), dtype=np.complex64)

    # Map each output frame back to a (fractional) position in the input
    for m_out in range(n_frames_out):
        m_src = m_out / stretch            # fractional frame index
        k = int(np.floor(m_src))           # left neighbour
        frac = m_src - k                   # interpolation weight

        if k >= Z.shape[1] - 1:            # tail-padding
            mag = np.abs(Z[:, -1])
            delta = dphi[:, -1]
        else:
            # Linear magnitude interpolation
            mag = (1 - frac) * np.abs(Z[:, k]) + frac * np.abs(Z[:, k + 1])
            delta = dphi[:, k]

        phi += delta                       # accumulate true phase
        Z_out[:, m_out] = mag * np.exp(1j * phi)

    # ------ synthesis ------
    _, y = istft(Z_out, fs=fs, window=win, nperseg=n_fft,
                 noverlap=n_fft - hop)

    # Normalise to -1 … 1 and convert to 16-bit PCM
    y /= np.max(np.abs(y) + 1e-8)
    pcm = (y * 32767).astype(np.int16)

    # Dump into an in-memory WAV file
    buf = io.BytesIO()
    wav_write(buf, fs, pcm)
    return buf.getvalue()


def naive_speed_change(x, speed: float, fs: int = 44_100) -> bytes:
    """
    Change playback speed by naive resampling (decimation / interpolation).

    Parameters
    ----------
    x      : np.ndarray
        1-D (mono) or 2-D (channels × samples) float audio in [-1, 1].
    speed  : float
        >1 speeds up (shorter, higher-pitched); <1 slows down (longer, lower-pitched).
    fs     : int, default 44100
        Sample rate of *x*.

    Returns
    -------
    bytes
        16-bit WAV file containing the resampled audio.
    """
    # -------- ensure mono and float32 ----------
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0)

    # -------- resample to new length ----------
    new_len = max(1, int(len(x) / speed))
    y = resample(x, new_len)          # FFT-based resampling

    # -------- normalise & convert to PCM ----------
    y /= np.max(np.abs(y) + 1e-8)
    pcm = (y * 32767).astype(np.int16)

    buf = io.BytesIO()
    wav_write(buf, fs, pcm)
    return buf.getvalue()

# ---------- helper: turn WAV bytes back into a mono float array ----------
def wav_bytes_to_array(wav_bytes):
    sr, data = wavfile.read(BytesIO(wav_bytes))
    # normalise to float32 in –1 … 1
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    if data.ndim == 2:                   # stereo → mono
        data = data.mean(axis=1)
    return sr, data


# ---------- spectrum helper ----------
def spectrum_db(x, fs):
    N      = len(x)
    X      = np.fft.rfft(x)
    freqs  = np.fft.rfftfreq(N, 1/fs)
    mags   = 20 * np.log10(np.abs(X) + 1e-12)  # dB magnitude
    return freqs, mags

original_audio = st.audio_input("Talk into it!")

if original_audio:

    if "previous_audio" not in st.session_state or st.session_state["previous_audio"] != original_audio:
        st.session_state["button1"] = False
        st.session_state["button2"] = False
        st.session_state["previous_audio"] = original_audio

    original_data = resample_audio(original_audio, 44100)
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    speed = st.slider("Choose your speed", min_value=0.25, max_value=4.0, value=1.0, step=0.25)
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    pv_audio = phase_vocoder(original_data, speed)
    naive_audio = naive_speed_change(original_data, speed)

    if st.button("Phase vocode my voice!", use_container_width=True):
        st.session_state["button1"] = not st.session_state["button1"]

    if st.session_state["button1"]:

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
            st.write("Phase vocoded 👍")
            st.audio(pv_audio, format="audio/wav")
        with col2:
            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
            st.write("Naive implementation 👎")
            st.audio(naive_audio, format="audio/wav")

        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)


        fs,  naive_arr = wav_bytes_to_array(naive_audio)
        _,   pv_arr    = wav_bytes_to_array(pv_audio)      # same fs as above

        f_orig, mag_orig   = spectrum_db(original_data, fs)
        f_pv,   mag_pv     = spectrum_db(pv_arr,        fs)
        f_naive, mag_naive = spectrum_db(naive_arr,     fs)

        
        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
        if st.button("How does it look like in the frequency domain?", use_container_width=True):
            st.session_state["button2"] = not st.session_state["button2"]

        if st.session_state["button1"] and st.session_state["button2"]:

            # --------- assemble long-format DataFrame for Altair ----------
            df = pd.DataFrame({
                "Frequency": np.concatenate([f_orig, f_pv, f_naive]),
                "Amplitude (dB)": np.concatenate([mag_orig, mag_pv, mag_naive]),
                "Version": (["Original"]        * len(f_orig)   +
                            ["Phase Vocoder"]  * len(f_pv)     +
                            ["Naive Resample"] * len(f_naive))
            })

            # --------- Altair chart ----------
            df_plot = df.query("20 <= Frequency <= 20000")

            chart = (
                alt.Chart(df_plot)
                .mark_line(clip=True)          # clip the path exactly at the plot edges
                .encode(
                    x=alt.X(
                        "Frequency:Q",
                        scale=alt.Scale(type="log", domain=[20, 2e4]),
                        title="Frequency (Hz)"
                    ),
                    y=alt.Y("Amplitude (dB):Q", title="Magnitude (dB)"),
                    color=alt.Color("Version:N", title="")
                )
                .properties(width="container", height=320)
            )
            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
            st.altair_chart(chart, use_container_width=True)

            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
            st.write("""The phase vocoder implementation is much more faithful to the original signal in the frequency domain.
                     Hope this page helps you understand the phase vocoder better and one day contribute back to the community! 💪""")



            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
            with st.expander("Here is my implementation of the phase vocoder in Python"):
                st.code("""
def phase_vocoder(x, speed: float, fs: int = 44_100,
                  n_fft: int = 2048, hop: int | None = None) -> bytes:
    if hop is None:
        hop = n_fft // 4

    # ------ mono-ise & make float32 ------
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0)

    # ------ analysis ------
    win = get_window("hann", n_fft, fftbins=True)
    _, _, Z = stft(x, fs=fs, window=win, nperseg=n_fft,
                   noverlap=n_fft - hop, boundary=None, padded=False)

    # How much we *stretch* the sound on the time axis.
    # speed 2.0 (twice as fast)  -> stretch 0.5 (half the frames)
    stretch = 1.0 / speed

    # Target number of frames in the output spectrogram
    n_frames_out = int(np.ceil(Z.shape[1] * stretch))

    # Phase-accumulator & expected phase advance between frames
    phi = np.angle(Z[:, 0])
    dphi = np.angle(Z[:, 1:]) - np.angle(Z[:, :-1])
    dphi = np.unwrap(dphi, axis=1)

    # Container for the time-scaled STFT
    Z_out = np.empty((Z.shape[0], n_frames_out), dtype=np.complex64)

    # Map each output frame back to a (fractional) position in the input
    for m_out in range(n_frames_out):
        m_src = m_out / stretch            # fractional frame index
        k = int(np.floor(m_src))           # left neighbour
        frac = m_src - k                   # interpolation weight

        if k >= Z.shape[1] - 1:            # tail-padding
            mag = np.abs(Z[:, -1])
            delta = dphi[:, -1]
        else:
            # Linear magnitude interpolation
            mag = (1 - frac) * np.abs(Z[:, k]) + frac * np.abs(Z[:, k + 1])
            delta = dphi[:, k]

        phi += delta                       # accumulate true phase
        Z_out[:, m_out] = mag * np.exp(1j * phi)

    # ------ synthesis ------
    _, y = istft(Z_out, fs=fs, window=win, nperseg=n_fft,
                 noverlap=n_fft - hop)

    # Normalise to -1 … 1 and convert to 16-bit PCM
    y /= np.max(np.abs(y) + 1e-8)
    pcm = (y * 32767).astype(np.int16)

    # Dump into an in-memory WAV file
    buf = io.BytesIO()
    wav_write(buf, fs, pcm)
    return buf.getvalue()
                """, language="python")

            




st.markdown('<div style="margin-top: 200px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")


