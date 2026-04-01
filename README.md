📧 Email Spam Detection Web App

A Machine Learning-based web application that classifies emails as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and a trained model.


⭐ Features
-----------------------------------------
🔍 Detects whether an email is Spam or Ham
🌐 User-friendly Flask web interface
⚡ Real-time prediction
🧠 Uses TF-IDF Vectorization + ML Model
💻 Built with Python, Flask, pandas and Scikit-learn


🛠️ Tech Stack
--------------------------------------------------
Frontend: HTML, CSS
Backend: Flask (Python)
Machine Learning: Scikit-learn
NLP: TF-IDF Vectorizer
Model Storage: Pickle


📂 Project Structure
-----------------------------------------------------

email-spam-detector/
│
│
├── templates/
│   └── index1.html (internal CSS)
│
│   ├── model1.pkl
│   └── vectorizer.pkl
│
├── app.py
├── requirements.txt
└── README.md


⚙️ How It Works
--------------------------------------------------------
User enters an email message in the web app
Input is sent to Flask backend
Text is converted into numerical features using TF-IDF
Trained ML model predicts:
Ham (Not Spam)
Spam
Result is displayed on the webpage
