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

    # Custom HTML and CSS for animated bubbles with names
    bubbles_html = """
    <style>
    .bubbles {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 50px;
    }
    .bubble {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 14px;
        color: white;
        margin: 0 10px;
        animation: changeColor 3s infinite;
    }
    @keyframes changeColor {
        0% { background-color: #FF5733; }
        25% { background-color: #33FF57; }
        50% { background-color: #3357FF; }
        75% { background-color: #FF33A1; }
        100% { background-color: #FF5733; }
    }
    </style>
    <div class="bubbles">
        <div class="bubble">Zaky</div>
        <div class="bubble">Komang</div>
        <div class="bubble">Agung</div>
        <div class="bubble">Niko</div>
        <div class="bubble">Rama</div>
    </div>
    """
    st.markdown(bubbles_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
