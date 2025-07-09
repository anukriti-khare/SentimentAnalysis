This project presents a complete end-to-end sentiment analysis pipeline built on a dataset of 1.6 million labeled tweets. The goal is to classify tweets as either **positive** or **negative** using classical machine learning techniques. The dataset underwent thorough preprocessing including lowercasing, URL and punctuation removal, stopword filtering, and lemmatization to clean noisy social media text.

For feature extraction, **TF-IDF Vectorization with unigrams and bigrams**, was used followed by training and evaluating three different models, **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear SVM**. Among these, Logistic Regression performed the best with an accuracy of approximately **79.5%** and a macro **F1-score of 0.80**, showing a well-balanced sentiment classification.

The project includes extensive exploratory analysis, visualization of word frequencies, bigrams, and tweet length distributions, as well as comparative performance metrics such as accuracy, classification reports, and confusion matrices for each model.

To make this project interactive and user-friendly, a **Streamlit web application** was developed where users can input custom tweet text and choose a model to predict sentiment in real-time. The app supports displaying predicted sentiment along with the model’s confidence.

**App Link:** https://huggingface.co/spaces/anukriti-khare/Sentiment-Analysis

