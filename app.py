import streamlit as st
import pickle

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OneHotEncoder

st.header("Banglore House Price Prediction Model")

model=pickle.load(open('model.pkl','rb'))
dataframe=pickle.load(open('dataframe.pkl','rb'))

location_name=dataframe['location'].unique()


def price_predict(location,total_sqft,bath,bhk):
    x={'location':location,'total_sqft':total_sqft,'bath':bath,'bhk':bhk}
    df=pd.DataFrame(x, index=[0])
    price=model.predict(df)
    return f'INR {round(price[0][0],3)} Lakhs'

location=st.selectbox('Please Select Location',(location_name))

total_sqft=st.number_input('Please select Total Sqft')

bath = st.slider('Please select bathroom', 0, 10, 1)

bhk = st.slider('BHK Size', 0, 20,2)


if st.button('Predict Price'):
    st.header(price_predict(location,total_sqft,bath,bhk))
