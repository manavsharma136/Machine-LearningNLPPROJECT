# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    if request.method=="POST":
        comment=request.form['comment']
        data=[comment]
        man=[]
        revi = re.sub('[^a-zA-Z]', ' ', str(data))
        revi = revi.lower()
        revi = revi.split()
        ps = PorterStemmer()
        revi = [ps.stem(word) for word in revi if not word in set(stopwords.words('english'))]
        revi = ' '.join(revi)
        man.append(revi)
        vect=cv.transform(man).toarray()
        my_prediction=classifier.predict(vect)
    return  render_template('result.html',prediction=my_prediction)
#    y_pred=classifier.predict(x_test)
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)

if __name__=='__main__':
    app.run(debug=True)
