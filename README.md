# Streamlit-Powered-AI-Diabetes-Risk-Predictor

## Overview
The Streamlit-Powered AI Diabetes Risk Predictor is an interactive web application designed to assess an individual's risk of developing diabetes. Leveraging machine learning techniques and the power of Streamlit, this tool provides an easy-to-use interface for users to input their health data and receive personalized risk assessments.

## Project Features
- **Interactive User Interface**: Built with Streamlit, allowing users to input their health information effortlessly.
- **AI-Driven Predictions**: Utilizes a trained Random Forest Classifier to predict diabetes risk based on user inputs.
- **Data Preprocessing**: Implements feature scaling and selection to ensure accurate predictions.
- **Model Persistence**: Uses `pickle` for saving and loading the trained model, scaler, and feature selector.

## Technical Details
### Libraries and Tools
- **Streamlit**: For building the interactive web application.
- **scikit-learn**: For machine learning, data preprocessing, and model evaluation.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **pyngrok**: For creating a public URL to access the Streamlit app on Google Colab.
- **pickle**: For serializing and deserializing the model, scaler, and selector.

### Data Preprocessing
1. **Scaling**: StandardScaler is used to normalize the features to have a mean of 0 and a standard deviation of 1.
2. **Feature Selection**: SelectKBest is employed to select the top features based on ANOVA F-value scores.

### Machine Learning Model
- **Model Used**: Random Forest Classifier
- **Training**: The model is trained on a diabetes dataset to predict the likelihood of developing diabetes based on various health metrics.
- **Evaluation**: Achieved an accuracy of 98.19% on the test set.

## Setup Instructions

### Requirements
- Python 3.x
- Streamlit
- scikit-learn
- pandas
- numpy
- pyngrok
