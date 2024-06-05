
import mlflow.pyfunc
import streamlit as st
import pandas as pd
import numpy as np


model_name = "SVM"
model = mlflow.pyfunc.load_model(f"models:/SVM/1")


st.title("Lung Cancer Prediction")
st.image("C:\mlflow\Globant_mlflow\LC.jpeg", caption="Lung Cancer Awareness")
st.markdown("""
This application uses a machine learning model to predict the likelihood of lung cancer based on several health factors.
Please enter the following details:
""")

# Define the features list
features = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
            'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
            'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
            'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

# Initialize an empty list to store user inputs
data = []

# Loop through each feature and collect user input
for feature in features:
    n = st.number_input(feature)
    data.append(n)

# Display the collected data
st.write("Collected data:", data)



if st.button("Predict"):

    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    
    if(prediction[0]==0):

        st.write(f"You are healthy")

    else:
        st.write(f"You have lungs cancer")


#footer

st.markdown("""
    <style>
        .footer {
            position: relative;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        <p>Developed by <a href="www.linkedin.com/in/yuvraj-singh-147905212" target="www.linkedin.com/in/yuvraj-singh-147905212">Yuvraj Singh</a></p>
    </div>
""", unsafe_allow_html=True)