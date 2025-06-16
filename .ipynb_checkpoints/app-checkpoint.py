import logging
import os
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
def load_model(model):
    with open(model, 'rb') as model:
        model = pickle.load(model)
    return model

# Function to make predictions
def make_prediction(model, input_data):
    return model.predict(input_data)

# Load the trained Random Forest model
model = 'model.pkl'
model_rf = load_model(model)

# Define the home page route
@app.route('/home')
def home():
    return render_template('./home.html')

@app.route('/index')
def prediction_page():
    return render_template('./index.html')

@app.route('/about')
def about():
    return render_template('./about.html')

@app.route('/contaminants')
def contaminants():
    return render_template('./contaminants.html')

@app.route('/guidelines')
def guidelines():
    return render_template('./guidelines.html')

@app.route('/addres')
def addres():
    return render_template('./addres.html')

@app.route('/feature')
def feature():
    return render_template('./featureimp.html')

@app.route('/std')
def std():
    return render_template('./std.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    user_input = []
    for param in ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']:
        value = float(request.form[param])
        user_input.append(value)
    input_data = np.array([user_input])

    # Make prediction
    prediction = make_prediction(model_rf, input_data)

    # Determine prediction result message
    if prediction[0] == 0:
        result_message = "The water sample does not seem to be potable 😓"
        
        # Further classify non-potable water
        if user_input[0] >= 6.5 and user_input[0] <= 7.5 and \
           user_input[1] < 160 and user_input[2] < 178.77 and \
           user_input[3] < 4 and user_input[4] < 192.12 and \
           user_input[5] >= 250 and user_input[5] <= 1000 and \
           user_input[8] >= 0 and user_input[8] <= 5:
               result_message += " It is suitable for agricultural use 🤝"
        elif user_input[0] >= 6 and user_input[0] <= 9 and \
             user_input[1] >= 100 and user_input[1] <= 500 and \
             user_input[2] >= 500 and user_input[2] <= 5000 and \
             user_input[3] < 1 and user_input[4] >= 250 and \
             user_input[4] <= 1000 and user_input[5] >= 100 and \
             user_input[5] <= 5000 and user_input[6] >= 0.1 and \
             user_input[6] <= 10 and user_input[7] < 10 and \
             user_input[8] >= 1 and user_input[8] <= 100:
                 result_message += " It is suitable for industrial use 🤝"
        else:
            result_message += " It is not safe for agricultural or industrial use 😓"
    else:
        result_message = "The water sample is safe to drink 🤝"
        
    # Render the result page with the prediction message
    return render_template('result.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)
