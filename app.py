import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
from tensorflow.keras.models import load_model
import seaborn as sns

st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

FEATURE_NAMES = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

FEATURE_DESC = {
    'CRIM': 'Per capita crime rate by town',
    'ZN': 'Proportion of residential land zoned for lots over 25,000 sq.ft.',
    'INDUS': 'Proportion of non-retail business acres per town',
    'CHAS': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
    'NOX': 'Nitric oxides concentration (parts per 10 million)',
    'RM': 'Average number of rooms per dwelling',
    'AGE': 'Proportion of owner-occupied units built prior to 1940',
    'DIS': 'Weighted distances to five Boston employment centres',
    'RAD': 'Index of accessibility to radial highways',
    'TAX': 'Full-value property-tax rate per $10,000',
    'PTRATIO': 'Pupil-teacher ratio by town',
    'B': '1000(Bk - 0.63)^2 where Bk is the proportion of black people by town',
    'LSTAT': '% lower status of the population'
}

st.title("Housing Price Prediction")
st.markdown("""
This app predicts housing prices using a deep learning model trained on the Boston Housing dataset.
Adjust the features on the sidebar and see how they impact the predicted price!
""")

def load_model_and_params():
    try:
        model = load_model('housing_regression_model.h5')
        if os.path.exists('model_params.pkl'):
            with open('model_params.pkl', 'rb') as f:
                params = pickle.load(f)
                mean = params['mean']
                std = params['std']
                train_metrics = params.get('train_metrics', None)
                test_metrics = params.get('test_metrics', None)
        else:
            (x_train, _), _, mean, std = load_data_for_app()
            train_metrics = test_metrics = None
        return model, mean, std, train_metrics, test_metrics
    except Exception as e:
        st.error(f"Error loading model: {e}. Please train the model first by running 'python regression_model.py'")
        st.stop()

def load_data_for_app():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return (x_train, y_train), (x_test, y_test), mean, std

col1, col2 = st.columns([2, 1])

with col1:
    try:
        model, mean, std, train_metrics, test_metrics = load_model_and_params()
        (x_train, y_train), (x_test, y_test), _, _ = load_data_for_app()

        st.header("Model Performance Metrics")

        if test_metrics:
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.subheader("Training Set Metrics")
                st.metric("MAE", f"{train_metrics['mae']:.4f}")
                st.metric("MSE", f"{train_metrics['mse']:.4f}")
                st.metric("RMSE", f"{train_metrics['rmse']:.4f}")
                st.metric("R² Score", f"{train_metrics['r2']:.4f}")
                st.metric("Model Accuracy", f"{train_metrics['accuracy_percentage']:.2f}%", 
                          help="Percentage of variance explained by the model (R²)")
            with metrics_col2:
                st.subheader("Test Set Metrics")
                st.metric("MAE", f"{test_metrics['mae']:.4f}")
                st.metric("MSE", f"{test_metrics['mse']:.4f}")
                st.metric("RMSE", f"{test_metrics['rmse']:.4f}")
                st.metric("R² Score", f"{test_metrics['r2']:.4f}")
                st.metric("Model Accuracy", f"{test_metrics['accuracy_percentage']:.2f}%",
                         help="Percentage of variance explained by the model (R²)")

            st.info("""
            Metrics Explained:
            - MAE: Mean Absolute Error - Average of absolute differences between predictions and actual values
            - MSE: Mean Squared Error - Average of squared differences between predictions and actual values
            - RMSE: Root Mean Squared Error - Square root of MSE, gives error in the same units as the target
            - R²: Coefficient of Determination - Proportion of variance in target explained by the model
            - Accuracy: R² expressed as a percentage - Higher is better
            """)

        st.header("Dataset Exploration")
        train_df = pd.DataFrame(x_train, columns=FEATURE_NAMES)
        train_df['PRICE'] = y_train
        st.subheader("Sample Data")
        st.dataframe(train_df.head())
        st.subheader("Feature Correlation")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(y_train, kde=True, ax=ax)
        ax.set_xlabel("Price (in $1000s)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        if os.path.exists('training_history.png'):
            st.subheader("Model Training History")
            st.image('training_history.png')
    except Exception as e:
        st.error(f"Error loading data or model: {e}")

with col2:
    st.header("Feature Inputs")
    st.write("Adjust the values to predict housing prices")
    feature_values = {}
    with st.form("prediction_form"):
        for i, feature in enumerate(FEATURE_NAMES):
            min_val = float(x_train[:, i].min() * std[i] + mean[i])
            max_val = float(x_train[:, i].max() * std[i] + mean[i])
            step = (max_val - min_val) / 100
            default_val = float(mean[i])
            feature_values[feature] = st.slider(
                f"{feature}", 
                min_value=min_val, 
                max_value=max_val,
                value=default_val,
                step=step,
                help=FEATURE_DESC.get(feature, "")
            )
        submit_button = st.form_submit_button(label="Predict Price")

    if submit_button:
        features = np.array([list(feature_values.values())])
        features = (features - mean) / std
        prediction = model.predict(features)[0][0]
        price = prediction * 1000
        st.header("Prediction")
        st.markdown(f"""
        ### Predicted House Price:
        # ${price:,.2f}
        """)
        st.subheader("Key Factors")
        st.info("Features with higher impact on the prediction:")
        last_layer_weights = model.layers[-2].get_weights()[0]
        feature_impact = {FEATURE_NAMES[i]: abs(last_layer_weights[i][0]) for i in range(len(FEATURE_NAMES))}
        top_features = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, impact in top_features:
            st.markdown(f"- **{feature}**: {FEATURE_DESC.get(feature, '')}")

st.markdown("---")
st.caption("Deep Learning Regression Model | Boston Housing Datase
