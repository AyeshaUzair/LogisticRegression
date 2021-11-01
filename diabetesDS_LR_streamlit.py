#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:19:57 2021

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
# import plotly.io as pio
# pio.renderers.default='browser'
# %matplotlib inline

st.title("Diabetes Dataset Modeling")
st.write("Select filters from sidebar to view more")
st.sidebar.title("Select an option:")
dpdwn = st.sidebar.selectbox("",[
                                "Prediction",
                                "Model Evaluation",
                                "Correlation Heatmap",
                                "Basic DataSet Visualizations",
                                "Logistic Regression Plot",
                                ])


df = pd.read_csv("Diabetes_dataset.csv")
print(df.info())
df.drop_duplicates()

# Change Outcome Datatype from int (1s and 0s) to Boolean (True and False)
# print(df['Outcome'].unique())
#df["Outcome"] = df["Outcome"].astype(bool)

# Initial Dataset Pandas Profiling
# profile = pp.ProfileReport(df)
# profile.to_file("Diabetese_dataset_EDA.html")


# =============================================================================
# # Copy dataset before alteration
# df1 = df.copy()
# 
# # Mean Calculations
# mean_insulin = df['Insulin'].mean()
# mean_glucose = df['Glucose'].mean()
# mean_bp = df['BloodPressure'].mean()
# mean_bmi = df['BMI'].mean()
# mean_skinthickness = df['SkinThickness'].mean()
# 
# # Deal with zeros in Skin thickness, BMI, BloodPressure, Glucose, Insulin
# # Convert 0s to nan and then fillna with mean
# df1['Insulin'] = df['Insulin'].mask(df['Insulin']==0).fillna(mean_insulin)
# df1['Glucose'] = df['Glucose'].mask(df['Glucose']==0).fillna(mean_glucose)
# df1['BloodPressure'] = df['BloodPressure'].mask(df['BloodPressure']==0).fillna(mean_bp)
# df1['BMI'] = df['BMI'].mask(df['BMI']==0).fillna(mean_bmi)
# df1['SkinThickness'] = df['SkinThickness'].mask(df['SkinThickness']==0).fillna(mean_skinthickness)
# print(df1.info())
# 
# # Change Outcome Datatype
# df1["Outcome"] = df1["Outcome"].astype(bool)
# 
# # Check new/cleaned dataset
# print(df1.info())
# 
# New Profiling Report
# profile_df1 = pp.ProfileReport(df1)
# profile_df1.to_file("Diabetese_dataset_cleaned.html")
# =============================================================================


# Variables
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# # Modeling
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.70, random_state=0)
# regressor = LogisticRegression()
# regressor.fit(X_train,y_train)
# # Pickling trained model for later use
# pickle.dump(regressor,open('model.pkl','wb'))

regressor = pickle.load(open('model.pkl','rb'))

# Model parameters
intercept = regressor.intercept_
coeffs = regressor.coef_
colls = X.columns
evals = []
for i in range(coeffs.shape[1]):
    evals.append([colls[i], coeffs[0][i]])
print(evals)
print("Coefficients: ", coeffs)
print("Variables: ", colls)
print("Intercept (expected mean value of Outcome when all variables are 0): ", intercept)


# Visualizing Regression Model
fig1 = sns.lmplot(x="Glucose", y="Outcome", data=df, logistic=True, y_jitter=.03)
# fig2 = sns.lmplot(x="Age", y="Outcome", data=df, logistic=True, y_jitter=.03)
# fig3 = sns.lmplot(x="Pregnancies", y="Outcome", data=df, logistic=True, y_jitter=.03)
# fig4 = sns.lmplot(x="BMI", y="Outcome", data=df, logistic=True, y_jitter=.03)
# fig5 = sns.lmplot(x="DiabetesPedigreeFunction", y="Outcome", data=df, logistic=True, y_jitter=.03)
# fig6 = sns.lmplot(x="Insulin", y="Outcome", data=df, logistic=True, y_jitter=.03)
# fig7 = sns.lmplot(x="BloodPressure", y="Outcome", data=df, logistic=True, y_jitter=.03)

# Correlation
corr = df.corr()
print(corr)

# figure22 = sns.jointplot(data = df, kind="scatter",x="DiabetesPedigreeFunction", y = "Pregnancies",hue="Outcome")

# fig8 = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cbar = True,cmap="rocket_r")
 
# Findings
finding1 = "No. of Pregnancies, Glucose levels, BMI and Age have strong correlation with the Outcome variable"
finding2 = "No. of Pregnancies, Glucose levels, Blood Pressure have a strong correlation with Age"
finding3 = "Age and Blood Pressure have a high correlation"
finding4 = "BMI and Skin Thickness have a very high correlation.. so on"


figure2 = px.histogram(df,x="Outcome")
figure3 = px.histogram(df, x='Pregnancies', y='Outcome')
figure5 = px.histogram(df, x='Age', y='Outcome')
figure6 = px.histogram(df, x='Glucose', y='Outcome')
# Predict from Test Dataset
y_pred = regressor.predict(X_test)

# Confusion Matrix
conf = confusion_matrix(y_test, y_pred)
cmtx = pd.DataFrame(
    confusion_matrix(y_test, y_pred), 
    index=['True: Yes', 'True: No'], 
    columns=['Predicted: Yes', 'Predicted: No']
)
print("confusion matrix: ", conf)
print(cmtx)

TP = conf[0][0]
FP = conf[0][1]
FN = conf[1][0]
TN = conf[1][1]

accuracy = (TP+TN) / (TP + TN + FN + TN)
print("accuracy: ", accuracy)

acc_score = accuracy_score(y_test, y_pred)
print(acc_score)

clasf_report = classification_report(y_test, y_pred)
print(clasf_report)


# coeff = list(regressor.coef_[0])
# labels = list(X.columns)
# features = pd.DataFrame()
# features['Features'] = labels
# features['importance'] = coeff
# features.sort_values(by=['importance'], ascending=True, inplace=True)
# features['positive'] = features['importance'] > 0
# features.set_index('Features', inplace=True)
# features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
# plt.xlabel('Importance')
# figure4 = px.histogram(features, x="importance", y=features.index, color = features.positive.map({True: 'blue', False: 'red'}))



if dpdwn == "Logistic Regression Plot":
    st.subheader("Logistic Regression Plot with Glucose as Independent Variables")
    st.pyplot(fig1)
    # st.pyplot(fig2)
    # st.pyplot(fig3)
    # st.pyplot(fig4)
    # st.pyplot(fig5)
    # st.pyplot(fig6)
    # st.pyplot(fig7)

        
if dpdwn == "Basic DataSet Visualizations":
    st.subheader("Outcome Count")
    st.plotly_chart(figure2)    
    st.subheader("Pregnancies vs Outcome Count")
    st.plotly_chart(figure3) 
    st.subheader("Age vs Outcome Count")
    st.plotly_chart(figure5) 
    st.subheader("Glucose vs Outcome Count")
    st.plotly_chart(figure6) 
    
    
if dpdwn == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cbar = True,cmap="rocket_r", ax=ax)
    st.write(fig)
    st.success(finding1)
    st.success(finding2)
    st.success(finding3)
    st.success(finding4)
    
if dpdwn == "Model Evaluation":
    st.header("Model Evaluation")
    st.write("**Dependent Variable:** Outcome")
    st.write("**Independent Variables:** [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]")
    st.subheader("Intercept")
    st.text(intercept)
    st.subheader("Variable Coefficients")
    for i in evals:
        st.text(i)
    st.subheader("Confusion Matrix")
    st.write(cmtx)
    st.subheader("Accuracy Score")
    st.write(acc_score)
    st.subheader("Classification Report")
    st.write(clasf_report)


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
    # Yes/No Prediction
    y_pred = regressor.predict(x_test)
    st.spinner()
    if st.button("Get Result"):
        if y_pred[0] == 0:
            st.subheader("Result: False")
            st.balloons()
            st.write("The patient does not seem to have diabetes.")
        else:
            st.subheader("Result: True")
            st.error("The patient seems to have diabetes.")
            st.write("The patient seems to have diabetes.")
    
        st.subheader("Prediction Probability")
        # Prediction probability
        pred1 = regressor.predict_proba(x_test) # False Case Probability | True Case Probability
        st.write("False Case Probability : {}".format(pred1[0][0]))
        st.write("True Case Probability : {}".format(pred1[0][1]))
    
