import streamlit as st
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

st.set_page_config(page_title="AQI Prediction App", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #212121;
        }
        .stButton>button {
            background-color: #6200EE;
            color: white;
            border-radius: 5px;
            font-size: 18px;
            padding: 12px;
        }
        .stButton>button:hover {
            background-color: #3700B3;
        }
        .stSlider>div>div>input {
            background-color: #333;
            color: white;
        }
        .stSelectbox>div>div>div {
            background-color: #333;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

def preprocess_input(PM2_5, PM10, NO2, CO, O3, SO2):
    return tf.constant([[PM2_5, PM10, NO2, CO, O3, SO2]], dtype=tf.float32)
st.title("🌍 Air Quality Index Prediction 🌿")
st.markdown("""
    This web app uses machine learning models to predict the **Air Quality Index (AQI)** based on input features such as PM2.5, PM10, NO2, CO, O3, and SO2.
    Select your desired model and input values to receive the predicted AQI and its classification.
""")
st.sidebar.title("🔧 Adjust Input Parameters")
st.sidebar.markdown("Adjust the values of the air quality parameters to predict the AQI of a specific city.")

PM2_5 = st.sidebar.slider('PM2.5 (µg/m³)', 0.0, 500.0, 25.0, 10.0, help="Particulate Matter less than 2.5 µm")
PM10 = st.sidebar.slider('PM10 (µg/m³)', 0.0, 500.0, 25.0, 10.0, help="Particulate Matter less than 10 µm")
NO2 = st.sidebar.slider('NO2 (ppb)', 0.0, 500.0, 25.0, 10.0, help="Nitrogen Dioxide level")
CO = st.sidebar.slider('CO (ppm)', 0.0, 40.0, 0.5, 0.5, help="Carbon Monoxide level")
O3 = st.sidebar.slider('O3 (ppb)', 0.0, 1000.0, 25.0, 10.0, help="Ozone concentration")
SO2 = st.sidebar.slider('SO2 (ppb)', 0.0, 2000.0, 25.0, 10.0, help="Sulfur Dioxide level")

input_data = preprocess_input(PM2_5, PM10, NO2, CO, O3, SO2)
model_names = ['LinearRegression', 'RandomForestRegressor', 'Neural Network']
model_selection = st.selectbox("🔍 Select Model", model_names, index=0)

model_paths = {
    'LinearRegression': './DL_Models/models/model_91.h5',
    'RandomForestRegressor': './DL_Models/models/model_169.h5',
    'Neural Network': './DL_Models/models/model_187.h5',
}

model = tf.keras.models.load_model(model_paths[model_selection])
if st.button("Predict AQI 📊"):
    prediction = model.predict(input_data)[0][0]
    st.subheader(f"Predicted AQI: {prediction:.2f}")
    
    color_ranges = {
        (0, 50): '#1FE140', 
        (51, 100): '#F5B700', 
        (101, 150): '#F26430', 
        (151, 200): '#DF2935', 
        (201, 300): '#D77A61', 
        (301, float('inf')): '#4D5061'
    }

    aqi_quality_table = {
        (0, 50): 'Good',
        (51, 100): 'Satisfactory',
        (101, 150): 'Moderate',
        (151, 200): 'Poor',
        (201, 300): 'Very Poor',
        (301, float('inf')): 'Severe'
    }

    prediction_color = 'black'
    aqi_quality = ''
    for range_, color in color_ranges.items():
        if range_[0] <= prediction <= range_[1]:
            prediction_color = color
            aqi_quality = aqi_quality_table[range_]
            break
    
    st.markdown(f"""
        <div style="background-color:{prediction_color}; padding:20px; border-radius:10px; text-align:center;">
            <h3 style="color:white; font-size:24px;">{aqi_quality}</h3>
            <h3 style="color:white; font-size:20px;">Predicted AQI: {prediction:.2f}</h3>
        </div>
    """, unsafe_allow_html=True)

st.subheader("📊 Detailed Graphical Insights")
input_names = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
input_values = [PM2_5, PM10, NO2, CO, O3, SO2]

graph_option = st.selectbox("Choose a graph to visualize:", [
    "Bar Chart of Parameters",
    "Line Plot of Inputs",
    "Heatmap of Correlations",
    "Pie Chart of Pollutants Share",
    "Boxplot of Value Distribution"
])

if graph_option == "Bar Chart of Parameters":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(input_names, input_values, color='lightblue')
    ax.set_title('Bar Chart - Air Quality Parameters')
    ax.set_ylabel('Concentration')
    st.pyplot(fig)

elif graph_option == "Line Plot of Inputs":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(input_names, input_values, marker='o', linestyle='-', color='lime')
    ax.set_title('Line Plot - Air Quality Parameters')
    ax.set_ylabel('Concentration')
    st.pyplot(fig)

elif graph_option == "Heatmap of Correlations":
    df = pd.DataFrame([input_values], columns=input_names)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Heatmap - Parameter Correlation')
    st.pyplot(fig)

elif graph_option == "Pie Chart of Pollutants Share":
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(input_values, labels=input_names, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    ax.set_title('Pie Chart - Pollutant Distribution')
    st.pyplot(fig)

elif graph_option == "Boxplot of Value Distribution":
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=pd.DataFrame([input_values], columns=input_names), ax=ax)
    ax.set_title('Boxplot - Value Distribution')
    st.pyplot(fig)

# Model Information
st.markdown("""
    ### 📚 Model Information:
    Each model corresponds to the number of trainable parameters. Models with more parameters tend to capture more complex patterns in the data.

    | Model Name                 | Description                       |
    |----------------------------|-----------------------------------|
    | **LinearRegression**       | A relatively simple model.        |
    | **RandomForestRegressor**  | A more complex model.             |
    | **Neural Network**         | The most complex of the three.    |
""")

st.markdown("---")

# Project Images
st.subheader("📸 Project Images")
image_folder = './DL_Models/images'
image_files = sorted(os.listdir(image_folder))
image_index = st.slider("🔄 Browse through images", 0, len(image_files) - 1, 0)
image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split('.')[0].replace('_', ' ').title()
st.image(image_path, caption=image_title, use_column_width=True)
