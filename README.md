📧 Email Spam Detection Web App
-----------------------------------------------------

A Machine Learning-based web application that classifies emails as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and a trained model.


⭐ Features
-----------------------------------------
🔍 Detects whether an email is Spam or Ham <br>
🌐 User-friendly Flask web interface <br>
⚡ Real-time prediction <br>
🧠 Uses TF-IDF Vectorization + ML Model <br>
💻 Built with Python, Flask, pandas and Scikit-learn <br>


🛠️ Tech Stack
--------------------------------------------------
Frontend: HTML, CSS <br>
Backend: Flask (Python) <br>
Machine Learning: Scikit-learn <br>
NLP: TF-IDF Vectorizer <br>
Model Storage: Pickle <br>


📂 Project Structure
-----------------------------------------------------

email-spam-detector/ <br>
│
│
├── templates/ <br>
│   └── index1.html (internal CSS) <br>
│
│   ├── model1.pkl <br>
│   └── vectorizer.pkl <br>
│
├── app.py <br>
├── requirements.txt <br>
└── README.md


⚙️ How It Works
--------------------------------------------------------
User enters an email message in the web app <br>
Input is sent to Flask backend <br>
Text is converted into numerical features using TF-IDF <br>  

Trained ML model predicts: <br>
Ham (Not Spam) <br>
Spam <br>
Result is displayed on the webpage <br>

The dataset is taken from kaggle !
