import streamlit as st
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloads (only needed the first time)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text preprocessing function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Prediction function
def predict_news(news_text):
    processed_text = preprocess_text(news_text)
    vectorized_input = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_input)
    return "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article below, and the model will predict whether it's real or fake.")

user_input = st.text_area("Paste the news content here:")

if st.button("Analyze"):
    if user_input.strip():
        result = predict_news(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text to analyze.")
