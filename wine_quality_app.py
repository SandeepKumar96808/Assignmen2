import streamlit as st
import numpy as np
import pickle
import sklearn

# Load the saved model
with open('rf_white_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to get user input
def get_user_input():
    # Use actual feature names and adjust the slider ranges as appropriate for each feature
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 7.0)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.1, 2.0, 0.5)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.3)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.0, 65.0, 5.0)
    chlorides = st.sidebar.slider('Chlorides', 0.01, 0.2, 0.05)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 80, 30)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6, 300, 150)
    density = st.sidebar.slider('Density', 0.98, 1.005, 0.995)
    pH = st.sidebar.slider('pH', 2.5, 4.0, 3.2)
    sulphates = st.sidebar.slider('Sulphates', 0.3, 2.0, 0.6)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 15.0, 10.5)

    # Create a numpy array with the inputs
    features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                         free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]).reshape(1, -1)

    return features

# Main function to display the app
def main():
    st.title("Wine Quality Prediction App")

    # Get user input
    user_input = get_user_input()

    # Prediction
    prediction = model.predict(user_input)
    
    # Display the prediction
    st.subheader('Prediction')
    st.write('The predicted quality of the wine is:', prediction[0])

if __name__ == '__main__':
    main()
