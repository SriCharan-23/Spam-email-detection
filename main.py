from flask import Flask,render_template,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)

# Loading model and vectorizer
model = pickle.load(open('model1.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Home route
@app.route("/")
def home():
    return render_template('index1.html')

# prediction route
@app.route('/predict',methods = ['POST'])
def predict():
        try:
            input_mail = request.form['E-mail']  # Taking input from user
            input_data = vectorizer.transform([input_mail]) # vectorizing the email, means converts text into numbers so ML model can easily understand

            prediction = model.predict(input_data)  # predicting whther the email is 'spam' or 'ham'

            result = "It is a Ham Mail" if prediction[0] == 1 else "It is a spam mail"

            return render_template('index1.html', prediction_text = f"prediction: {result}" )
        
        except ValueError:
             
             return render_template('index1.html', prediction_text = "Invalid input entered !")
        
if __name__ == "__main__":
     app.run(debug=True)
             


