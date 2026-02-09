# 📩 SMS Spam Classifier

A **Machine Learning–based SMS Spam Detection web app** built with **Python, scikit-learn, and Streamlit**.  
The application classifies text messages as **Spam** or **Ham (Not Spam)** in real time and allows users to adjust **spam sensitivity using a threshold slider**.

---
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Model-orange)

## 🚀 Live Demo
🔗 **Streamlit App:**  
https://sms-spam-classifier-j.streamlit.app/

---

## 🎯 Features
- ✅ Real-time SMS spam detection
- 🎚 Adjustable **Spam Sensitivity Threshold**
- 📊 Spam confidence level with progress bar
- 🎨 Clean UI with color-coded result cards
- ⚡ Fast predictions using trained ML model
- 🌐 Deployed on **Streamlit Community Cloud (Free)**

---

## 🧠 Machine Learning Models
Models evaluated during training:
- **Multinomial Naive Bayes (Best performing)**
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- XGBoost

Final model selection was based on **precision**, which is critical for spam detection to minimize false positives.

---

## 🗂 Project Structure
```text
sms-spam-classifier/
│
├── app.py              # The main Streamlit application
├── model.pkl           # Pre-trained Classification Model (e.g., Naive Bayes)
├── vectorizer.pkl      # Pre-trained TF-IDF/Count Vectorizer
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Files to exclude from Git 
```
---

## ⚙️ Tech Stack
- **Python 3**
- **Streamlit**
- **scikit-learn**
- **Pandas & NumPy**
- **TF-IDF / CountVectorizer**
- **Matplotlib / Seaborn (EDA & evaluation)**

---

## ▶️ Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/gauravrajput4/sms-spam-classifier.git
cd sms-spam-classifier
```
