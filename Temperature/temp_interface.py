from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and label encoder
clf = joblib.load('temp_logistic_regression_model.pkl')
label_classes = np.load('temp_label_classes.npy',allow_pickle=True)

def get_diagnosis(temperature):
    # Make predictions on the input temperature
    temperature_array = np.array([[temperature]])
    prediction_encoded = clf.predict(temperature_array)
    prediction_decoded = label_classes[prediction_encoded]
    return prediction_decoded[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        diagnosis = get_diagnosis(temperature)
        return render_template('result.html', temperature=temperature, diagnosis=diagnosis)

if __name__ == '__main__':
    app.run(debug=True)
