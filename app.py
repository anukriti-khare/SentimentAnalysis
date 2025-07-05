import streamlit as st
import joblib
import numpy as np

# Load vectorizer and models
tfidf = joblib.load('tfidf_vectorizer.pkl')
logistic_model = joblib.load('logistic_regression_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Title and description
st.title("Tweet Sentiment Analysis")
st.write("Predict sentiment using TF-IDF + ML Models")

# User input
user_input = st.text_area("Enter Tweet Text:")

# Model selection
model_choice = st.selectbox("Choose Model", ("Logistic Regression", "Naive Bayes", "SVM"))

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Vectorize input
        vect_input = tfidf.transform([user_input])

        # Select model
        if model_choice == "Logistic Regression":
            pred = logistic_model.predict(vect_input)[0]
            proba = logistic_model.predict_proba(vect_input)[0]
        elif model_choice == "Naive Bayes":
            pred = nb_model.predict(vect_input)[0]
            proba = nb_model.predict_proba(vect_input)[0]
        else:
            pred = svm_model.predict(vect_input)[0]
            # SVM does not have predict_proba by default, so confidence is not available
            proba = None

        sentiment = "Positive" if pred == 1 else "Negative"
        st.write(f"**Predicted Sentiment:** {sentiment}")

        if proba is not None:
            confidence = np.max(proba) * 100
            st.write(f"**Confidence:** {confidence:.2f}%")
        else:
            st.write("Confidence score not available for this model.")
