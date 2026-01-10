import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (run once)
nltk.download('stopwords')

# Load model & vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing tools
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# App UI
st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Spam Detection App")

st.write("Enter a message below to check whether it is *Spam* or *Not Spam*.")

user_input = st.text_area("Message", height=150)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess input
        text = user_input.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stopwords_set]
        clean_text = " ".join(words)

        # Prediction
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")

# Footer
st.markdown("---")
st.caption("Built using NLP, Naive Bayes & Streamlit")