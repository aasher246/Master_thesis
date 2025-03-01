from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and label classes
model = joblib.load('RR_RandomForest_model.pkl')
label_classes = np.load('bp_label_classes.npy', allow_pickle=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            rr_value = float(request.form['respiration_rate'])
            if rr_value < 0:
                raise ValueError("Respiration rate cannot be negative.")
            prediction = model.predict([[rr_value]])
            predicted_category = label_classes[prediction[0]]
            return render_template('result.html', respiration_rate=rr_value, category=predicted_category)
        except ValueError as e:
            return render_template('index.html', error=str(e))
        except Exception as e:
            return render_template('index.html', error="An unexpected error occurred: " + str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
