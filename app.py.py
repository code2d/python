# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# Load the dataset
df = pd.read_csv(r"C:\Users\cs998\PRG1\placement.csv")

# Features and labels
X = df[['cgpa']]
y = df['placed']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.header("WELCOME TO SAPTHAGIRI PLACEMENT CELL" ,divider='rainbow')
st.subheader("Student Placement Prediction")
img = Image.open(r"C:\Users\cs998\PRG1\PLACEMENT-CELL.png")
st.image(
    img ,
    width=800,
    )

# User input for CGPA
cgpa = st.number_input("Please Select Your CGPA", min_value=1.0, max_value=10.0, value=7.0)

# Make prediction
prediction = model.predict([[cgpa]])

# Display the result
st.subheader("Result")
if prediction[0] == 1:
    st.header('Congratulations you have been placed :blue[] :sunglasses:')
else:
    st.header('Sorry student is not yet placed. Lets hope for best. 	:blush: ')

st.write('This website was created by ECE Students 	:zap:')
st.write('Thank u for visting')
             
