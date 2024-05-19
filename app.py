import os
import nltk

# Specify the path to NLTK data
nltk.data.path.append('C:\\Users\\punya\\AppData\\Roaming\\nltk_data')

nltk.download('movie_reviews')
nltk.download('punkt')


import nltk
nltk.download('movie_reviews')
nltk.download('punkt')  # for tokenizing

from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import random

app = Flask(__name__)

# Download the movie reviews dataset
nltk.download('movie_reviews')

# Load and preprocess the movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Extract the 2000 most common words as features
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[f'contains({word})'] = (word in document_words)
    return features

# Create feature sets for the classifier
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]

# Convert the feature sets to a format suitable for Scikit-learn
vec = DictVectorizer()
X_train, y_train = zip(*train_set)
X_train = vec.fit_transform(X_train)

X_test, y_test = zip(*test_set)
X_test = vec.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    features = document_features(nltk.word_tokenize(text))
    features = vec.transform(features)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
