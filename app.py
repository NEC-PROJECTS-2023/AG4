from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    train_data = []
    data_files = ['Youtube01-Psy.csv', 'Youtube02-KatyPerry.csv','Youtube03-LMFAO.csv', 'Youtube04-Eminem.csv', 'Youtube05-Shakira.csv']
    for file in data_files:
        data = pd.read_csv(file)
        train_data.append(data)
    data_frame = pd.concat(train_data)
    df_data=data_frame[["CONTENT","CLASS"]]
    df_x=df_data['CONTENT']
    df_y=df_data.CLASS 
    corpus =df_x
    cv=CountVectorizer()
    X=cv.fit_transform(corpus)
    X_train, X_test, y_train, y_test= train_test_split(X, df_y, test_size=0.33,random_state=42)
    clf=BernoulliNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    if request.method =='POST':
        comment =request.form['comment']
        data=[comment]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result.html',prediction=my_prediction)
    
if __name__=='__main__':
    app.run(debug=True)