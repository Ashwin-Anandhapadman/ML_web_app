from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__) #we will use this application variable to deploy the model in AWS console 
app = application

'''
In AWS console deployment:
option_settings:
    "aws:elasticbeanstalk:container:python":
        WSGIPath: application: application

        #the first application is the file name and the second one is the flask intialization
'''




# Import ridge regressor and standard scaler pickle files
ridge_model = pickle.load(open('models/ash_ridge.pkl', 'rb'))
scaler_model = pickle.load(open('models/ash_scaler.pkl', 'rb'))

@app.route("/") 
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST']) 
def predict():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
            
            # Prepare data for prediction (reshape into 2D array)
            new_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            new_data_scaled = scaler_model.transform(new_data)
            
            # Predict the result
            result = ridge_model.predict(new_data_scaled)
            
            return render_template('home.html', results=result[0])
        except Exception as e:
            return str(e)  # Simple error message for debugging
    else:
        return render_template("home.html")


if __name__ == "__main__": 
    app.run(host="0.0.0.0")
