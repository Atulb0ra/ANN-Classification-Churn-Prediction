import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
import streamlit as st

#Loading the model
model = tf.keras.models.load_model('models.h5')

#load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#streamlit app
st.title('Customer churn prediction')

CreditScore=st.number_input('Credit Score')
Geography=st.selectbox('Geography', onehot_encoder_geo.categories_[0])
Gender=st.selectbox('Gender', label_encoder_gender.classes_)
Age=st.slider('Age', 18, 92)
Tenure=st.slider('Tenure', 0, 10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('NumOfProduct', 1, 4)
HasCrCard=st.selectbox('Has Credit Card', [0,1])
IsActiveMember=st.selectbox('Is Active Member', [0,1])
EstimatedSalary=st.number_input('Estimated Salary')

input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})


#one-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine one-hot encode coulmns with input data
input_df = pd.concat([input_data.drop('Geography', axis = 1), df], axis=1)

#Scaling the input data
input_scaled = scaler.transform(input_df)

## predict churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]


st.write(f'Churn Probability :{prediction_proba:.2f} ')
if(prediction_proba > 0.5):
    st.write("The customer is likely to churn")

else:
    st.write("The customer is not likely to churn")