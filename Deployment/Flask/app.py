import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, url_for

app=Flask(__name__)
model=pickle.load(open('sv.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method=='POST':
        sl=float(request.form['Sepal Length'])
        sw = float(request.form['Sepal Width'])
        pl = float(request.form['Petal Length'])
        pw = float(request.form['Petal Width'])
    f=[np.array([pl,pw,sl,sw])]
    p=model.predict(f)
    return render_template('result.html', prediction_text=p[0])

if __name__== '__main__':
    app.run(debug=True)