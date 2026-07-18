import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
label_map_inv = {0: 'negative', 1: 'neutral', 2: 'positive'}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'<.*?>', ' ', str(text))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

st.title("Customer Sentiment Analyzer")
st.write("Enter a review below to classify it as positive, neutral, or negative.")

user_input = st.text_area("Review text", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            cleaned = preprocess(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            label = label_map_inv[pred]
            emoji = {'positive': '😀', 'neutral': '😐', 'negative': '🙁'}[label]
            st.subheader(f"Prediction: {label.capitalize()} {emoji}")
            st.write(f"Confidence: {max(proba):.2%}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
