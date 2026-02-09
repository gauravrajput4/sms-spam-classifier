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
    padding: 22px;
    border-radius: 14px;
    margin-top: 20px;
    font-size: 16px;
}


.ham {
    background-color: #052e1c;      
    color: #dcfce7;                 
    border-left: 6px solid #22c55e;
}

/* SPAM CARD */
.spam {
    background-color: #3f0a0a;    
    color: #fee2e2;                
    border-left: 6px solid #ef4444;
}


.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #e5e7eb;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📩 Spam Message Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Check whether a message is Spam or Ham</div>", unsafe_allow_html=True)

st.write("")


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