import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a function for Prediction
def rockmine_prediction(input_data):
    try:
        # Split the string by commas and convert to float
        input_list = [float(i) for i in input_data.split(',')]
        input_array = np.array(input_list)

        # Check if the input data can be reshaped to the required dimensions (1, 60)
        if input_array.shape[0] != 60:
            st.error("Input data must contain exactly 60 values.")
            return None

        # Reshape the array for the model
        input_data_reshaped = input_array.reshape(1, -1)

        # Make a prediction
        prediction = loaded_model.predict(input_data_reshaped)

        # Return the prediction result
        if prediction[0] == 'R':
            return 'The object is a Rock'
        else:
            return 'The object is a Mine'
    except ValueError:
        st.error("Invalid input. Please enter a comma-separated list of numbers.")
        return None

def main():
    # Create a title
    st.title('Rock vs Mine Prediction')

    # Text area to input the data
    input_data = st.text_area("Sonar Signal Value Input")
    
    # Code for Prediction
    result = ''

    # Create a button for starting prediction
    if st.button('Rock or Mine?'):
        result = rockmine_prediction(input_data)
    
    if result:
        st.success(result)

if __name__ == '__main__':
    main()
