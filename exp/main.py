import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import os

st.markdown("""
    <style>
        .main { 
            background-color: #121212;
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #181818;
            color: white;
        }
        .stButton>button {
            background-color: #6200ea;
            color: white;
            font-size: 18px;
            padding: 18px 50px;
            text-align: center;
            border: none;
            cursor: pointer;
            border-radius: 50px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s, background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3700b3;
            transform: translateY(-4px);
        }
        .stTitle {
            font-size: 45px;
            color: #ff9800;
            margin-top: 20px;
            text-align: center;
        }
        .stHeader {
            font-size: 40px;
            color: #ff9800;
            margin-top: 20px;
            text-align: center;
        }
        .stSidebar {
            background-color: #121212;
        }
        .stSelectbox select {
            background-color: #333333;
            color: white;
            border-radius: 8px;
            padding: 12px;
            font-size: 18px;
        }
        .stSlider div, .stNumberInput div {
            color: white;
        }
        .stMarkdown {
            font-size: 20px;
            color: #e0e0e0;
        }
        .stTextInput input {
            color: white;
            background-color: #333333;
            border-radius: 8px;
        }
        .stImage {
            margin: 30px 0;
        }
        .stTextInput {
            margin: 20px 0;
        }
        .stPlotlyChart {
            margin: 30px 0;
        }
        .stSelectbox {
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

st.title('🌟 Enhanced AQI Prediction App')
st.write("""
This web app uses advanced machine learning models to predict the Air Quality Index (AQI) based on **PM2.5** and **PM10**.  
Choose a model from the drop-down menu, input data, and see real-time predictions.
""")
scaler = pickle.load(open('exp/scaler.pkl', 'rb'))
model_option = st.selectbox(
    'Select Model',
    ['Model 1', 'Model 2', 'Model 3']
)

if model_option == 'Model 1':
    model = pickle.load(open('exp/model_no_1.pkl', 'rb'))
elif model_option == 'Model 2':
    model = pickle.load(open('exp/model_no_2.pkl', 'rb'))
else:
    model = pickle.load(open('exp/model_no_3.pkl', 'rb'))


st.sidebar.title('Input Parameters')
PM2_5 = st.sidebar.number_input('PM2.5 (µg/m³)', min_value=0.0, max_value=500.0, value=20.0, step=10.0)
PM10 = st.sidebar.number_input('PM10 (µg/m³)', min_value=0.0, max_value=500.0, value=20.0, step=10.0)


input_data = {'PM2.5': PM2_5, 'PM10': PM10}
data_df = pd.DataFrame([input_data], columns=['PM2.5', 'PM10'])
data_scaled = scaler.transform(data_df)

col1, col2, col3 = st.columns([1, 2, 1])
col2.subheader('Click to Predict')
if col2.button('Get AQI Prediction'):
    prediction = model.predict(data_scaled)[0]
    st.subheader(f'Predicted AQI: {prediction}')

    if prediction <= 50:
        st.write("AQI Category: **Good**")
    elif prediction <= 100:
        st.write("AQI Category: **Moderate**")
    elif prediction <= 150:
        st.write("AQI Category: **Unhealthy for Sensitive Groups**")
    elif prediction <= 200:
        st.write("AQI Category: **Unhealthy**")
    elif prediction <= 300:
        st.write("AQI Category: **Very Unhealthy**")
    else:
        st.write("AQI Category: **Hazardous**")

    fig = go.Figure(data=[
        go.Bar(name='PM2.5', x=['AQI'], y=[PM2_5], marker_color='blue'),
        go.Bar(name='PM10', x=['AQI'], y=[PM10], marker_color='orange')
    ])
    fig.update_layout(
        title="Comparison of PM2.5 and PM10",
        xaxis_title="AQI Prediction",
        yaxis_title="Concentration (µg/m³)",
        template="plotly_dark",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig)

    trend_fig = go.Figure(data=[
        go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[prediction, prediction - 20, prediction + 10, prediction - 30, prediction + 15],
            mode='lines+markers',
            name='AQI Trend',
            line=dict(color='cyan', width=4)
        )
    ])
    trend_fig.update_layout(
        title="AQI Prediction Trend Over Time",
        xaxis_title="Time (Arbitrary Units)",
        yaxis_title="Predicted AQI",
        template="plotly_dark",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(trend_fig)

image_folder = './exp/images'
image_files = sorted(os.listdir(image_folder))
image_index = st.slider("Slide to select an image", 0, len(image_files) - 1, 0)

image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split('.')[0].replace('_', ' ').title()
st.title(image_title)
st.image(image_path, use_column_width=True)

st.markdown('---')