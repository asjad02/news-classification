from flask import Flask, render_template, url_for, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=['POST'])
def predict():
   
    vocabulary_load =joblib.load("count_vector")
    count_vector = CountVectorizer(vocabulary = vocabulary_load)
    # training_data = count_vector.fit_transform(X_train)
    # testing_data = count_vector.transform(X_test)

    model_pkl = open('model.pkl', 'rb')
    model = pickle.load(model_pkl)
    # vocabulary_to_load = pickle.load(open('count_vector', 'rb'))
    # loaded_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,
    #                                     ngram_size), min_df=1, vocabulary=vocabulary_to_load)
    # loaded_vectorizer._validate_vocabulary()
    
    id_to_category_load = np.load('id_to_category.npy').item()

    if request.method == 'POST':
        article = request.form['article']
        data = [article]
        vect = count_vector.transform(data)
        my_prediction = model.predict(vect)
        my_prediction = my_prediction.tolist()
        prediction =  id_to_category_load[my_prediction[0]]

    return render_template("result.html", prediction = prediction)


if __name__ == '__main__':
    app.run()