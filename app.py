import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# PAGE CONFIG (TOP BAR)
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# MAIN HEADING (INSIDE APP)
st.title("📩 Spam Detection App")
st.write("Enter a message below to check whether it is **Spam** or **Not Spam**.")

# DOWNLOAD NLTK DATA
nltk.download("stopwords")

# LOAD MODEL & VECTORIZER 
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# PREPROCESSING TOOLS
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words("english"))

# USER INPUT
user_input = st.text_area("Message", height=150)

# PREDICTION
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Text preprocessing
        text = user_input.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stopwords_set]
        clean_text = " ".join(words)

        # Vectorize & predict
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")

# FOOTER
st.markdown("---")
st.caption("Built using NLP, Naive Bayes & Streamlit")
