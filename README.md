

# AQI Prediction using Deep Learning & Machine Learning


## 1. Introduction

### 1.1 Purpose
The AQI Prediction system aims to forecast the Air Quality Index (AQI) based on pollutant concentrations in the air. Using machine learning models, this project predicts AQI values to help mitigate the harmful effects of air pollution by providing timely forecasts. The predictions will enable better planning and decision-making for public health, government bodies, and environmental organizations.

### 1.2 Scope
This project is designed to:
- Use historical air quality data to train machine learning models.
- Provide real-time AQI predictions using an interactive Streamlit frontend.
- Compare various machine learning models and choose the best-performing one.

---

## 2. Project Overview

### 2.1 Problem Statement
In urban areas, air pollution is a major concern, and accurate prediction of AQI can help in early detection of high pollution levels. This system automates the prediction of AQI based on historical data, allowing users to receive accurate forecasts and take preventive measures accordingly.

### 2.2 Objectives
- **Model Training**: Use various machine learning algorithms to train models on historical data.
- **Prediction Interface**: Implement a user-friendly, real-time prediction tool via Streamlit.
- **Model Optimization**: Tune model parameters and compare multiple models to ensure the best predictive performance.

---

## 3. System Design

### 3.1 High-Level Architecture

The AQI Prediction system is divided into the following components:
1. **Data Collection**: Historical AQI data is gathered from air quality monitoring stations.
2. **Data Preprocessing**: Data cleaning, feature selection, and normalization are done to prepare data for training.
3. **Model Training**: Various machine learning models (Linear Regression, Random Forest, Neural Networks, XGBoost) are trained and validated.
4. **Real-Time Prediction**: The model is deployed via Streamlit for real-time AQI predictions.
5. **Evaluation**: Model performance is assessed using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

---

## 4. File Structure

```
AQI Prediction using Deep Learning & Machine Learning/
├── src/                    # Deep Learning Model Files
│   ├── aqi_predictor_nn_final.ipynb
│   ├── images/
│   ├── main.py
│   └── models/
├── data/                          # Data Files
│   ├── city_aqi_day.csv           # Raw daily AQI data
│   ├── city_hour.csv.zip          # Raw hourly AQI data
│   ├── clean_data.csv             # Cleaned AQI data for training
│   └── no_missing.csv             # Data with no missing values
├── exp/                           # Experimentation Files (for model experimentation)
│   ├── aqi_predictor_exp.ipynb    # Experimental model code
│   ├── images/
│   ├── main.py
│   └── objects/
├── frontend/                      # Streamlit Frontend for Real-Time Predictions
│   └── main.py
├── .gitignore                     # Ignore unnecessary files
├── README.md                      # Project Documentation
└── requirements.txt               # Dependencies for the project
```

---

## 5. Preprocessing

The preprocessing phase includes several crucial steps to ensure that the data is ready for model training:

### 5.1 Data Cleaning
- **Missing Data**: Missing values in the dataset are handled either by **imputation** (using mean/median) or **removal** of rows/columns with missing values.
- **Outlier Detection**: Identifying and removing outliers that could negatively affect model performance.

### 5.2 Feature Selection
- **Redundant Features Removal**: We analyze the correlation between features and remove highly correlated or irrelevant features.
- **Dimensionality Reduction**: If needed, techniques like PCA (Principal Component Analysis) are applied to reduce the number of features while preserving important information.

### 5.3 Data Normalization
- **Scaling**: Features are normalized using Min-Max Scaling or Standard Scaling to ensure that the models are not biased toward any particular feature.

### 5.4 Data Splitting
- **Train-Test Split**: The dataset is split into **training** and **testing** sets (typically 80% for training and 20% for testing) to evaluate model performance.

---

## 6. Training

### 6.1 Model Initialization
In this phase, different models are initialized based on the nature of the data and the prediction task. The following models are used:
- **Linear Regression**: A baseline model to predict AQI based on a linear relationship between features.
- **Random Forest Regressor**: An ensemble method that builds multiple decision trees for more accurate predictions.
- **Neural Networks**: Deep learning models that can capture complex non-linear relationships between features.
- **XGBoost**: A high-performance gradient boosting model that is highly effective for regression tasks.

### 6.2 Model Training
Each model is trained on the training set using the following steps:
- **Fitting the model**: The model learns from the training data by adjusting its internal parameters to minimize the error.
- **Hyperparameter Tuning**: Models are fine-tuned by experimenting with different hyperparameters (e.g., number of trees in Random Forest, learning rate in Neural Networks).

### 6.3 Cross-Validation
- **K-Fold Cross-Validation**: The dataset is divided into **K** subsets, and each model is trained **K** times, each time using a different subset for validation, and the rest for training. This ensures the model generalizes well.

---

## 7. Optimization

Optimization is performed to fine-tune the models and improve their predictive power.

### 7.1 Hyperparameter Tuning
- **GridSearchCV**: A method that systematically works through multiple hyperparameter combinations to find the best set of parameters for each model.
- **RandomizedSearchCV**: A faster alternative to GridSearchCV where random combinations of hyperparameters are tested.

### 7.2 Feature Engineering
- **New Feature Creation**: Based on domain knowledge, new features might be created, such as interaction terms or polynomial features, to improve model performance.

### 7.3 Model Evaluation
After training, models are evaluated using various metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual AQI values.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the MSE to give error in the same units as the AQI.
- **R-squared**: A metric that shows how well the model explains the variance in the data.

---

## 8. Purpose of the `exp/` Directory

The `exp/` directory contains files used for experimentation with different models, configurations, and training techniques. These files serve to:

- **Test different models and approaches** before finalizing the best-performing one.
- **Validate model performance** through experiments.
- **Refine and compare results** of various models.
- Files in the `exp/` directory are not intended for production use and are used solely for exploratory and validation purposes. They are not executed in the production pipeline but are critical for model selection and fine-tuning.

---

## 9. Requirements

### 9.1 Software Requirements
- Python 3.8 or higher
- Jupyter Notebook or Jupyter Lab (for model training)
- Streamlit (for frontend)
- Required Python Libraries:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow (for deep learning models)
  - xgboost
  - matplotlib, seaborn (for data visualization)
  - streamlit (for frontend deployment)

### 9.2 Hardware Requirements
- CPU: Intel Core i5 or higher
- RAM: 8 GB or higher
- Disk Space: 2 GB for datasets and model files

---

## 10. Installation

### 10.1 Setting Up the Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ss1910singh/Air-Quality-Index-Deep-Learning-Machine-Learning-.git
   cd Air-Quality-Index-Deep-Learning-Machine-Learning
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```
   or
   ```bash
   conda create -n myenv python=3.10
   conda activate myenv
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Data (if not available locally)**
   - Data can be sourced from government or environmental monitoring APIs or repositories.

---

## 11. Running the Project

### 11.1 Training the Models
To train and evaluate models, run the following Jupyter notebook:
```bash
jupyter notebook DL_Models/aqi_predictor_nn_final.ipynb
```

### 11.2 Running the Streamlit Frontend
To start the real-time prediction interface:
```bash
streamlit run frontend/main.py
```
This will launch the application in your browser.

---

## 12. Evaluation Metrics

The model performance will be evaluated based on:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual AQI values.
- **Root Mean Squared Error (RMSE)**: Provides a measure of error in the same units as the AQI.
- **R-squared**: Indicates how well the model explains the variance in the target variable.

---

## 13. Conclusion

The AQI prediction system serves as an important tool for predicting air quality, helping individuals and authorities make informed decisions. By utilizing advanced machine learning techniques, the system offers accurate, real-time predictions of AQI, promoting public health awareness.

---
