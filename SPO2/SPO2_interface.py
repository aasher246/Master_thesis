from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the Random Forest model and label encoder
clf = joblib.load('spo2_random_forest_model.pkl')
label_classes = np.load('spo2_label_classes.npy', allow_pickle=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the SpO2 value from the form
            spo2_value = float(request.form['spo2'])
            spo2_array = np.array([[spo2_value]])

            # Make prediction
            prediction_encoded = clf.predict(spo2_array)
            prediction_decoded = label_classes[prediction_encoded][0]

            return render_template('result.html', spo2=spo2_value, prediction=prediction_decoded)
        
        except Exception as e:
            return render_template('index.html', message=f'An error occurred: {str(e)}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
