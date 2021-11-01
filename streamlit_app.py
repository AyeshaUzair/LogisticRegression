#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:45:28 2021

@author: ayeshauzair
"""

import pandas as pd
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, confusion_matrix, accuracy_score, classification_report
import math
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import plotly_express as px
# import plotly.graph_objs as go
# import plotly.subplots as sp
import streamlit as st
import plotly.io as pio
pio.renderers.default='browser'
# %matplotlib inline



st.title("Diabetes Dataset Modeling")

# Getting Pickled model 
regressor = pickle.load(open('model.pkl','rb'))


st.subheader("Prediction")
preg = st.number_input("Enter the number of pregnancies: ")
glucose = st.number_input("Enter the glucose level: ")
bp = st.number_input("Enter the blood pressure level: ")
skin_t = st.number_input("Enter the skin thickness: ")
insulin = st.number_input("Enter the insulin level: ")
bmi = st.number_input("Enter the bmi: ")
dpf = st.number_input("Enter the diabetes pedigree function: ")
age = st.number_input("Enter the age in years: ")

x_test = [[preg, glucose, bp, skin_t, insulin, bmi, dpf, age]]
# Yes/No Prediction
y_pred = regressor.predict(x_test)
if st.button("Get Result"):
    if y_pred[0] == 0:
        st.subheader("Result: False")
        st.balloons()
        st.success("The patient does not seem to have diabetes.")
    else:
        st.subheader("Result: True")
        st.error("The patient seems to have diabetes.")

    st.subheader("Prediction Probability")
    # Prediction probability
    pred1 = regressor.predict_proba(x_test) # False Case Probability | True Case Probability
    st.write("False Case Probability : {}".format(pred1[0][0]))
    st.write("True Case Probability : {}".format(pred1[0][1]))

