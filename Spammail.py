import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pickle

raw_maildata = pd.read_csv("spam.csv", encoding="latin-1")
raw_maildata = raw_maildata[['v1', 'v2']]  # Selecting only required columns and ignoring the columns which are not required
raw_maildata.columns = ['label', 'message']  # naming the columns
# print(raw_maildata.head())

# Label encoding
raw_maildata['label'] = raw_maildata['label'].map({'ham': 1, 'spam': 0})

x = raw_maildata['message'] 
y = raw_maildata['label']

# splitting the dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# This creates a TF-IDF vectorizer object
vectorizer = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
# what each parameter does is:
# TfidfVectorizer
# Converts text into numbers (feature vectors) so ML models can understand it.
# Uses TF-IDF (Term Frequency - Inverse Document Frequency).
# It is basically a NLP (Natural Language Processing) technique which proides scores for specific words in any mail
# like "Congratulations!", "Hurry up!" etc.,

# min_df=1
# ----------------
# Keeps words that appear in at least 1 document.
# If you set min_df=2, it would ignore rare words.

# stop_words='english'
# --------------------------
# Removes common words like:
# the, is, in, and, to
# These words don’t add much meaning.

# lowercase=True
# ----------------------
# Converts all text to lowercase:
# "Hello" → "hello"
x_train_feature = vectorizer.fit_transform(x_train)
# fit()
# Learns vocabulary from x_train
# Example: words like "free", "win", "offer"
# transform()
# Converts each message into a numeric vector

 # Output: x_train_feature → matrix of numbers
x_test_features = vectorizer.transform(x_test)
# Only transforms, does NOT learn again
# Uses the same vocabulary learned from training data


# Creating model
model = MultinomialNB()
model.fit(x_train_feature,y_train)  # Training data

# checking training accuracy
training_predict = model.predict(x_train_feature)
print(f"Accuracy of training data is: {accuracy_score(y_train,training_predict)}")

# checking testing accuracy
testing_predict = model.predict(x_test_features)
print(f"Accuracy of testing data is: {accuracy_score(y_test,testing_predict)}")

# Actual test data prediction and knowing accuracy
y_pred = model.predict(x_test_features)
print(f"Accuracy of model is: {accuracy_score(y_test,y_pred)}")




pickle.dump(model, open('model1.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
