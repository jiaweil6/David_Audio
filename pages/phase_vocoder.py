import streamlit as st
import streamlit.components.v1 as components
from sidebar import sidebar
import io
import soundfile as sf
import librosa

st.set_page_config(
    page_title="Phase Vocoder",
    page_icon="📽️",
    layout="centered",
    initial_sidebar_state="auto",
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

st.image("images/sample-waveform.gif", caption="Demo from Google Deepmind", use_container_width="always")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

st.write("""
Let's say we want to speed up the audio by 2x, means we want the length to be half the original. We can simply skip every other sample where the audio is decimated by 2.
Now if we just play the audio using the original sampling rate, we will have the same audio but twice as fast right? Let's visualize it to see if that makes sense!
""")

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

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

left, middle, right = st.columns(3)

# Easy mode
if left.button("Easy mode", icon="🍼", use_container_width=True):
    st.write("You clicked the plain button.")

# Medium mode
if middle.button("Medium mode", icon="😉", use_container_width=True):
    st.write("You clicked the emoji button.")

# GOATED MODE
if right.button("GOATED MODE", icon="🔥", use_container_width=True):
    st.write("You clicked the Material button.")

st.markdown('<div style="margin-top: 200px;"></div>', unsafe_allow_html=True)

st.image("images/icon.png", use_container_width="always")
