import string
import numpy as np
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
    return " ".join(y)

# Page config
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Load model & vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Custom CSS for cards
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
}
.ham {
    background-color: #e6f4ea;
    border-left: 6px solid #1a7f37;
}
.spam {
    background-color: #fdecea;
    border-left: 6px solid #d1242f;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>📩 Spam Message Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Check whether a message is Spam or Ham</div>", unsafe_allow_html=True)

st.write("")

# Input box
user_input = st.text_area(
    "✍️ Enter your message below",
    height=120,
    placeholder="Type your message here..."
)

if st.button("🔍 Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Vectorize
        input_vector = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(input_vector)[0]

        # Probability (Spam class = 1)
        if hasattr(model, "predict_proba"):
            spam_prob = model.predict_proba(input_vector)[0][1]
        else:
            spam_prob = 0.0

        confidence = int(spam_prob * 100)

        # Result card
        if prediction == 1:
            st.markdown(
                f"""
                <div class="card spam">
                    <h3>🚨 Spam Message</h3>
                    <p>This message looks like <b>SPAM</b>.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="card ham">
                    <h3>✅ Ham Message</h3>
                    <p>This message looks like <b>NOT SPAM</b>.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Confidence bar
        st.write("")
        st.write("📊 **Spam Confidence Level**")
        st.progress(confidence / 100)
        st.write(f"Confidence: **{confidence}%**")