import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import streamlit as st

##load the model
model=tf.keras.models.load_model('model.h5')

##load encoders and scaler

with open('OHE.pkl','rb') as f:
    onehot_encoder=pickle.load(f)

with open('label.pkl','rb') as f:
    label_encoder=pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

##streamlit app
st.title('customer churn prediction')

##userinput
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

##prepare i/p 
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender":[gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

#one hot encode the Geography column
ip_geo=onehot_encoder.transform([[geography]]).toarray()
ip_geo_df=pd.DataFrame(ip_geo,columns=onehot_encoder.get_feature_names_out(['Geography']))

#label encode for the gender column
input_data['Gender']=label_encoder.transform(input_data['Gender'])
#concating one hot encoder cols(df_geo) and df
df=pd.concat([input_data.drop('Geography',axis=1),ip_geo_df],axis=1)

#scale the data
scaled_data=scaler.transform(df)
scaled_data

##predicting churn
pred=model.predict(scaled_data,verbose=0)

st.write(f'churn probability={pred[0][0]: .2f}')

if pred[0][0]>0.5:
    st.write('the person is likey to leave')
else:
    st.write('the person is not likey leave')

