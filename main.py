import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()

max_len = 500
max_features = 10000

model = load_model('simple_rnn_imdb.h5')


def preprocess_text(text):
    words = text_to_word_sequence(text)

    encoded_review = []

    for word in words:
        index = word_index.get(word)

        if index is None or index >= max_features:
            encoded_review.append(2)
        else:
            encoded_review.append(index + 3)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=max_len
    )

    return padded_review


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input, verbose=0)

    score = float(prediction[0][0])

    sentiment = 'Positive' if score >= 0.5 else 'Negative'

    return sentiment, score


st.title('🎬 IMDB Movie Review Sentiment Analysis')

st.write(
    'Enter a movie review to classify it as positive or negative.'
)

user_input = st.text_area('Enter Movie Review')

if st.button('Classify Sentiment'):

    if user_input.strip():

        sentiment, score = predict_sentiment(user_input)

        if sentiment == 'Positive':
            st.success(f'Sentiment: {sentiment}')
        else:
            st.error(f'Sentiment: {sentiment}')

        st.metric('Prediction Score', f'{score:.4f}')

        st.progress(score)

    else:
        st.warning('Please enter a movie review.')