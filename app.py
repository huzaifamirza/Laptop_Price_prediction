import streamlit as st
import numpy as np
import pickle

# importing model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
st.title('Laptop Price Predictor')

# Brand selector
Company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
Type = st.selectbox('Type', df['TypeName'].unique())

# Ram
Ram = st.selectbox('Ram', df['Ram'].unique())

# Weight
weight = st.number_input('Weight of the Laptop')

# TouchScreen
Touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])

# IPS
IPS = st.selectbox('IPs', ['Yes', 'No'])


# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440','2304x1440'])

# Cpu
Cpu = st.selectbox('Cpu', df['Cpu Brand'].unique())

# HDD
HDD = st.selectbox('HDD', [0, 128, 256, 512, 1024, 2048])

# SDD
SDD = st.selectbox('SDD', [0, 8, 128, 256, 512, 1024])

# Gpu
Gpu = st.selectbox('Gpu', df['Gpu brand'].unique())

# Os
Os = st.selectbox('Os', df['Os'].unique())

if st.button('Predict Price'):
    ppi = None
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0
    if IPS == 'Yes':
        IPS = 1
    else:
        IPS = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = np.array([Company, Type, Ram, weight, Touchscreen, IPS, ppi, Cpu, HDD, SDD, Gpu, Os])

    query = query.reshape(1, 12)

    st.title(np.exp(pipe.predict(query)))



