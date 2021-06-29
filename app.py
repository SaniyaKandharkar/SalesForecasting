# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:24:47 2021

@author: saniya
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    database = "prediction_df"

    if request.method == 'POST':
        data1 = request.form('Date')
        data2 = request.form('Original Sales')
        arr = np.array([data1, data2])
        pred = model.predict(arr)
    
    
    
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(prediction_df)

    #output = prediction_df

    return render_template('index.html', prediction_text='Predicted sales are $ {}', data = pred)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    #output = prediction_df
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)