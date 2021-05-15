from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
from datetime import timedelta
from datetime import date
import datetime
import pandas as pd
import random
from random import randint
import numpy as np

import pickle

# load the model from disk
loaded_model=pickle.load(open('random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('real_2018.csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    date_now = date.today()
    return render_template('result.html',prediction = my_prediction,now=datetime.datetime.now())

@app.route('/val',methods=['POST'])
def val():
    return render_template('predict.html')

@app.route('/result',methods=['POST'])
def result():
	df=pd.read_csv('real_2018.csv')
	int_features=[x for x in request.form.values() if x!='']
	final_features=[np.array(int_features)]
	prediction=loaded_model.predict(df.iloc[:,:-1].values)
	num = random.randint(0,50)
	output=prediction[num]
	return render_template('predict.html',prediction_text='AQI is: {}'.format(output))



if __name__ == '__main__':
	app.run(debug=True)
