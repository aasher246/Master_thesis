from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the trained model and label encoder
clf = joblib.load('blood_pressure_model.pkl')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('bp_label_classes.npy', allow_pickle=True)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        bp_values = request.form['bp_values']
        systolic, diastolic = map(float, bp_values.split('/'))
        input_data = [[systolic, diastolic]]
        prediction = clf.predict(input_data)
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        return render_template('result.html', systolic=systolic, diastolic=diastolic, predicted_class=predicted_class)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
