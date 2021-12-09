from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import numpy as np
import pickle
import socket
from model.model import M
# from ml_model.model import MyModel
app = Flask(__name__)

model = torch.load("/common/users/shared/cs543_fall21_group6/project_2/flask_app/model/model_outgoing_flights.pt")
model = model.eval()
model = model.cpu()

with open('/common/users/shared/cs543_fall21_group6/project_2/origin_encoder_2.pkl', 'rb') as infile:
    origin_encoder = pickle.load(infile)
    
# with open('/common/users/shared/cs543_fall21_group6/project_2/counts_scaler.pkl', 'rb') as infile:
#     counts_scaler = pickle.load(infile)


@app.route('/')
def home():
   return render_template("index.html")


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return(render_template('index.html'))
    
    if request.method == 'POST':
        # Extract the input
        month = int(request.form['month'])
        date = int(request.form['date'])
        airport = request.form['airport']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[month, date, airport]],
                                       columns=['Month', 'DayofMonth', 'Origin'])
        input_variables['Origin'] = origin_encoder.transform(input_variables['Origin'].astype(str))
        input_variables = input_variables.to_numpy()
        input_variables = torch.from_numpy(input_variables)

        # Get the model's prediction
        prediction = np.round(model(input_variables).detach().numpy()[0,0])
        # prediction = counts_scaler.inverse_transform(prediction.detach().numpy())
        # counts_scaler.mean_, counts_scaler.var_
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return render_template('index.html',
                                     original_input={'Month':month,
                                                     'Date':date,
                                                     'Origin':airport},
                                     result=prediction
                                     )


if __name__ == '__main__':
    app.run(host=socket.gethostname(), port=9995)