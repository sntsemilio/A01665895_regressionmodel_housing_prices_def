import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

FEATURE_NAMES = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

FEATURE_DESC = {
    'MedInc': 'Median income in block group',
    'HouseAge': 'Median house age in block group',
    'AveRooms': 'Average rooms per household',
    'AveBedrms': 'Average bedrooms per household',
    'Population': 'Block group population',
    'AveOccup': 'Average household occupancy',
    'Latitude': 'Block group latitude',
    'Longitude': 'Block group longitude'
}

st.title("🏠 California Housing Price Prediction")
st.markdown("""
This app predicts housing prices using a regression model trained on the California Housing dataset.
Adjust the features below and see how they impact the predicted price!
""")

@st.cache_resource
def load_model_and_params():
    try:
        with open('A01665895_regressionmodel_housing_prices_cali.pkl', 'rb') as f:
            model = pickle.load(f)
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
        st.error(f"Error loading model: {e}. Please train and export the pickle model first!")
        st.stop()

def load_data_for_app():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    data = fetch_california_housing()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=113
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
        train_df = pd.DataFrame(x_train, columns=FEATURE_NAMES)
        train_df['PRICE'] = y_train

        st.header("Model Performance Metrics")

        if test_metrics:
            metrics_col1, metrics_col2 = st.columns(2)

            with metrics_col1:
                st.subheader("Training Set Metrics")
                st.metric("MAE", f"{train_metrics['mae']:.4f}")
                st.metric("MSE", f"{train_metrics['mse']:.4f}")
                st.metric("RMSE", f"{train_metrics['rmse']:.4f}")
                st.metric("MAPE", f"{train_metrics['mape']:.2f}%")
                st.metric("R² Score", f"{train_metrics['r2']:.4f}")
                st.metric("Model Accuracy", f"{train_metrics['accuracy_percentage']:.2f}%")

            with metrics_col2:
                st.subheader("Test Set Metrics")
                st.metric("MAE", f"{test_metrics['mae']:.4f}")
                st.metric("MSE", f"{test_metrics['mse']:.4f}")
                st.metric("RMSE", f"{test_metrics['rmse']:.4f}")
                st.metric("MAPE", f"{test_metrics['mape']:.2f}%")
                st.metric("R² Score", f"{test_metrics['r2']:.4f}")
                st.metric("Model Accuracy", f"{test_metrics['accuracy_percentage']:.2f}%")

            st.info("""
            **Metrics Explained:**
            - **MAE**: Mean Absolute Error
            - **MSE**: Mean Squared Error
            - **RMSE**: Root Mean Squared Error
            - **MAPE**: Mean Absolute Percentage Error
            - **R²**: Coefficient of Determination
            - **Accuracy**: R² expressed as a percentage
            """)

        st.header("Dataset Exploration")
        st.subheader("Sample Data")
        st.dataframe(train_df.head())

        st.subheader("Summary Statistics")
        st.dataframe(train_df.describe())

        st.subheader("Feature Correlation")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(y_train, kde=True, ax=ax)
        ax.set_xlabel("Price (in $100,000s)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        if os.path.exists('training_history.png'):
            st.subheader("Model Training Results")
            st.image('training_history.png')
    except Exception as e:
        st.error(f"Error loading data or model: {e}")

with col2:
    st.header("Feature Inputs")
    st.write("Adjust the values to predict housing prices")
    try:
        x_original = x_train * std + mean
        df_features = pd.DataFrame(x_original, columns=FEATURE_NAMES)

        feature_values = {}
        with st.form("prediction_form"):
            for i, feature in enumerate(FEATURE_NAMES):
                min_val = float(df_features[feature].min())
                max_val = float(df_features[feature].max())
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
            features_scaled = (features - mean) / std
            prediction = model.predict(features_scaled)
            price = prediction[0] * 100000 if np.ndim(prediction) == 1 else prediction[0][0] * 100000

            st.header("Prediction")
            st.markdown(f"""
            ### Predicted House Price:
            # ${price:,.2f}
            """)

            st.subheader("Key Factors")
            st.info("Features with higher impact on this prediction:")

            feature_correlations = abs(train_df.corr()['PRICE'])[:-1]
            top_features = feature_correlations.nlargest(5)

            for feature, importance in top_features.items():
                current_value = feature_values[feature]
                mean_value = mean[FEATURE_NAMES.index(feature)]

                if current_value > mean_value:
                    direction = "higher than average"
                    effect = "increases" if importance > 0 else "decreases"
                else:
                    direction = "lower than average"
                    effect = "decreases" if importance > 0 else "increases"

                st.markdown(f"- **{feature}**: {FEATURE_DESC.get(feature, '')} is {direction}, which generally {effect} the price")
    except Exception as e:
        st.error(f"Error setting up prediction interface: {e}")

st.markdown("---")
st.caption("Regression Model | California Housing Dataset | Last updated: 2025-06-02 | developed by @sntsemilio")