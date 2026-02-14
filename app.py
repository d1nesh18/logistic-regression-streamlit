import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("Diabetes Prediction — Logistic Regression")

df = pd.read_csv("diabetes (6).csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

st.header("Enter Patient Details")

inputs = []
for col in X.columns:
    val = st.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    inputs.append(val)

if st.button("Predict"):
    user_scaled = scaler.transform([inputs])
    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]

    if pred == 1:
        st.error(f"Diabetic — probability {prob:.2f}")
    else:
        st.success(f"Not Diabetic — probability {1-prob:.2f}")
