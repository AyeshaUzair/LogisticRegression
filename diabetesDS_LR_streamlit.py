#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:19:57 2021

@author: ayeshauzair
"""

import pandas as pd
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, confusion_matrix, accuracy_score, classification_report
import math
import pickle

# import pandas_profiling as pp
# from matplotlib import pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objs as go
# import plotly.subplots as sp
import streamlit as st
import plotly.io as pio
pio.renderers.default='browser'
# %matplotlib inline

st.title("Diabetes Dataset Modeling")
st.write("Select filters from sidebar to view more")
st.sidebar.title("Select a Plot")
dpdwn = st.sidebar.selectbox("",[
                                "1",
                                "2",
                                "3",
                                "Prediction"
                                ])


df = pd.read_csv("Diabetes_dataset.csv")
# profile = pp.ProfileReport(df)
# profile.to_file("Diabetese_dataset_EDA.html")
print(df.info())
df.drop_duplicates()


# # Mean Calculations
# mean_insulin = df['Insulin'].mean()
# mean_glucose = df['Glucose'].mean()
# mean_bp = df['BloodPressure'].mean()
# mean_bmi = df['BMI'].mean()
# mean_skinthickness = df['SkinThickness'].mean()


# # Copy dataset before alteration
# df1 = df.copy()


# # Deal with zeros in Skin thickness, BMI, BloodPressure, Glucose, Insulin
# # Convert 0s to nan and then fillna with mean
# df1['Insulin'] = df['Insulin'].mask(df['Insulin']==0).fillna(mean_insulin)
# df1['Glucose'] = df['Glucose'].mask(df['Glucose']==0).fillna(mean_glucose)
# df1['BloodPressure'] = df['BloodPressure'].mask(df['BloodPressure']==0).fillna(mean_bp)
# df1['BMI'] = df['BMI'].mask(df['BMI']==0).fillna(mean_bmi)
# df1['SkinThickness'] = df['SkinThickness'].mask(df['SkinThickness']==0).fillna(mean_skinthickness)
# print(df1.info())


# # Change Outcome Datatype
# df1["Outcome"] = df1["Outcome"].astype(bool)


# # Check new/cleaned dataset
# print(df1.info())


# New Profiling Report
# profile_df1 = pp.ProfileReport(df1)
# profile_df1.to_file("Diabetese_dataset_cleaned.html")


# Variables
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Modeling
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.70, random_state=0)
regressor = LogisticRegression()
regressor.fit(X_train,y_train)


# Model parameters
print("coef_: ", regressor.coef_)
print("intercept: ", regressor.intercept_)


# =============================================================================
# # Correlation
# corr = df1.corr()
# print(corr)
# 
# =============================================================================
figure22 = sns.jointplot(data = df, kind="scatter",x="DiabetesPedigreeFunction", y = "Pregnancies",hue="Outcome")

# =============================================================================
# figure1 = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cbar = True,cmap="rocket_r")
# 
# =============================================================================
finding1 = "No. of Pregnancies, Glucose levels, BMI and Age have strong correlation with the Outcome variable"
finding2 = "No. of Pregnancies, Glucose levels, Blood Pressure have a strong correlation with Age"
finding3 = "Age and Blood Pressure have a high correlation"
finding4 = "BMI and Skin Thickness have a very high correlation.. so on"


# =============================================================================
# figure2 = px.histogram(df1,x="Outcome")
# 
# =============================================================================

# Test
new_x = [[4,100,70,8,16,30,0.3,40]]


# Prediction probability
pred1 = regressor.predict_proba(new_x) # False Case Probability | True Case Probability
print("x1 ", pred1)

# Yes/No Prediction
pred2 = regressor.predict(new_x) 
print("x2 ", pred2)

y_pred = regressor.predict(X_test)
conf = confusion_matrix(y_test, y_pred)
print("confusion matrix: ", conf)

TP = conf[0][0]
FP = conf[0][1]
FN = conf[1][0]
TN = conf[1][1]

print(conf[0][0])
print(conf[1][0])
print(conf[0][1])
print(conf[1][1])

accuracy = (TP+TN) / (TP + TN + FN + TN)
print("accuracy: ", accuracy)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


coeff = list(regressor.coef_[0])
labels = list(X.columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
print(features.index)
# figure3 = features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
figure3 = px.histogram(features, x="importance", y=features.index, color = features.positive.map({True: 'blue', False: 'red'}))

#figure3.plt.xlabel('Importance')



if dpdwn == "1":
    st.subheader("Histogram: Gender Analysis")
    st.pyplot(figure22)
    st.write(accuracy)
        
if dpdwn == "2":
    st.subheader("Histogram: Age Group Analysis")
    # st.plotly_chart(figure2)    
    st.write(conf)
    
if dpdwn == "3":
    st.subheader("Histogram: Age Group vs Gender")
    st.plotly_chart(figure3)
    st.write(y_pred)
    
if dpdwn == "Prediction":
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
    y_pred = regressor.predict(x_test)
    
    if st.button("Get Result"):
        if y_pred[0] == 0:
            st.subheader("Result: False")
            st.success()
            st.balloons()
            st.write("The patient does not seem to have diabetes.")
        else:
            st.subheader("Result: True")
            st.error("The patient seems to have diabetes.")
            st.write("The patient seems to have diabetes.")
    
    
    
