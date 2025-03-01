from flask import Flask, render_template, request
import scipy.io
from io import BytesIO
import numpy as np
from matplotlib import pyplot as plt
import base64
import sys
import pickle
import logging
import joblib

sys.path.append('/Users/aashika/Desktop/Classes/Semester-3/Thesis/PythonFiles/Heartrate')
from Functions_HR import pre_processing, get_features

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def flatten_features(features_dict):
    feature_list = []
    for key in sorted(features_dict.keys()):  # Sort keys to preserve order
        value = features_dict[key]
        if isinstance(value, (list, np.ndarray)):
            feature_list.extend(np.ravel(value))  # Flatten any nested lists or arrays
        else:
            feature_list.append(value)
    return np.array(feature_list)

# Load models and label encoders
def load_model_and_labels(model_path, label_path):
    try:
        clf = joblib.load(model_path)
        label_classes = np.load(label_path, allow_pickle=True)
        return clf, label_classes
    except Exception as e:
        logging.exception(f'Failed to load model or label classes from {model_path} or {label_path}')
        return None, None

temp_clf, temp_label_classes = load_model_and_labels('temp_random_forest_model.pkl', 'temp_label_classes.npy')
bp_clf, bp_label_classes = load_model_and_labels('blood_pressure_model.pkl', 'bp_label_classes.npy')
rr_clf, rr_label_classes = load_model_and_labels('RR_RandomForest_model.pkl', 'rr_label_classes.npy')
spo2_clf, spo2_label_classes = load_model_and_labels('spo2_random_forest_model.pkl', 'spo2_label_classes.npy')

def get_diagnosis(clf, label_classes, input_data):
    prediction_encoded = clf.predict(input_data)
    prediction_decoded = label_classes[prediction_encoded]
    return prediction_decoded[0]

# Define dictionaries
weights = {
    'heart_rate': 15,
    'blood_pressure': 20,
    'body_temperature': 10,
    'respiratory_rate': 10,
    'spo2': 15
}

scores = {
    'heart_rate': {
        'Sinus Rhythm': 100,
        'Sinus Bradycardia': 75,
        'Sinus Tachycardia': 75,
        'Atrial Fibrillation': 50
    },
    'blood_pressure': {
        'hypotension': 75,
        'normal': 100,
        'elevated': 75,
        'hypertension_stage_1': 50,
        'hypertension_stage_2': 25,
        'hypertensive_crisis': 10
    },
    'body_temperature': {
        'low temperature': 50,
        'normal': 100,
        'elevated temperature': 75,
        'moderate fever': 50,
        'high fever': 25,
        'very high fever': 10
    },
    'respiratory_rate': {
        'Normal': 100,
        'Bradypnoea': 75,
        'Tachypnoea': 75
    },
    'spo2': {
        'Normal Blood Oxygen levels': 100,
        'Concerning Blood Oxygen levels': 75,
        'Low Blood Oxygen levels': 50,
        'Low Blood Oxygen levels that can affect your brain': 25,
        'Cyanosis': 10
    }
}

recommendations = {
    'heart_rate': {
        'Sinus Rhythm': 'Maintain current lifestyle. Make sure to include regular exercise, balanced diet, and stress management to keep your heart rhythm normal.',
        'Sinus Bradycardia': 'Bradycardia is when the heart rhythm is slower than normal. If asymptomatic, monitor regularly. If experiencing symptoms like dizziness or fatigue, consult a healthcare provider for further evaluation.',
        'Sinus Tachycardia': 'Tachycardia is when the heart rhythm is faster than normal. Reduce caffeine and alcohol intake, manage stress through relaxation techniques, and consult a healthcare provider if the condition persists.',
        'Atrial Fibrillation': 'Seek medical advice to manage and treat atrial fibrillation. Follow prescribed medication regimen and consider lifestyle changes like reducing alcohol consumption and increasing physical activity.'
    },
    'blood_pressure': {
        'hypotension': 'For Hypotension, increase fluid and salt intake under medical supervision, avoid sudden position changes, and consult a healthcare provider if symptoms persist.',
        'normal': 'Blood pressure normal, avoid excessive salt and alcohol to keep blood pressure within the normal range.',
        'elevated': 'Elevated BP can put you at risk. Adopt a heart-healthy diet, reduce sodium intake, increase physical activity, and monitor blood pressure regularly.',
        'hypertension_stage_1': 'To treat 1st stage Hypertension, you require some lifestyle modifications including a DASH diet, regular exercise, weight loss if overweight, and regular monitoring of blood pressure. Consult a healthcare provider for potential medication.',
        'hypertension_stage_2': 'To treat 2nd stage Hypertension, you will need to follow a strict regimen of prescribed antihypertensive medication, make significant lifestyle changes, and schedule regular follow-ups with a healthcare provider.',
        'hypertensive_crisis': 'Seek immediate medical attention to treat Hypertensive Crisis. This condition requires urgent treatment to prevent serious complications..'
    },
    'body_temperature': {
        'low temperature': 'Stay warm and hydrated, avoid exposure to cold, and seek medical advice if body temperature remains low.',
        'normal': 'Monitor Body temperature regularly to ensure it stays in the normal range',
        'elevated temperature': 'For elevated temperature, take rest, stay hydrated, and monitor for other symptoms. Seek medical advice if the temperature persists or other symptoms develop.',
        'moderate fever': 'For a moderate fever, take rest, stay hydrated, take antipyretics like acetaminophen or ibuprofen as needed, and consult a healthcare provider if the fever persists.',
        'high fever': 'Seek medical attention, as high fever can indicate a serious infection or other health issues. Ensure adequate hydration and rest.',
        'very high fever': 'Seek immediate medical attention; very high fever can be life-threatening and requires urgent treatment.'
    },
    'respiratory_rate': {
        'Normal': 'To maintain normal Respiration rate, avoid smoking, and ensure good air quality in your environment.',
        'Bradypnoea': 'Bradypnoea is when the breathing rate is lower than normal. Seek medical evaluation to determine the underlying cause. Avoid medications that can depress respiration and monitor for symptoms like dizziness or fatigue.',
        'Tachypnoea': 'Tachypnoea is when the breating rate is higher than normal.Identify and manage underlying causes such as infections, asthma, or anxiety. Seek medical advice if the condition persists.'
    },
    'spo2': {
        'Normal Blood Oxygen levels': 'Continue maintaining a healthy lifestyle, avoid smoking, and ensure regular physical activity to keep oxygen levels optimal.',
        'Concerning Blood Oxygen levels': 'Oxygen levels are concerning. Monitor regularly, increase oxygen intake by breathing deeply, and consult a healthcare provider for further evaluation.',
        'Low Blood Oxygen levels': 'Seek medical attention to identify and treat the underlying cause of low blood oxygen. Use supplemental oxygen if prescribed by a healthcare provider.',
        'Low Blood Oxygen levels that can affect your brain': 'Seek immediate medical attention, as severely low oxygen levels can lead to serious complications. Follow prescribed oxygen therapy and treatment plans.',
        'Cyanosis': 'Seek urgent medical attention, as cyanosis indicates critically low oxygen levels in the blood. Immediate intervention is necessary.'
    }
}

def calculate_health_score(diagnoses):
    total_score = 0
    total_weight = 0
    detailed_recommendations = {}

    for parameter, diagnosis in diagnoses.items():
        score = scores[parameter][diagnosis]
        weight = weights[parameter]
        total_score += score * weight
        total_weight += weight
        detailed_recommendations[parameter] = recommendations[parameter][diagnosis]

    health_score = total_score / total_weight if total_weight > 0 else 0

    return health_score, detailed_recommendations

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            # Handle file uploads
            if 'files' not in request.files:
                return render_template('index.html', message='Please upload both .hea and .mat files for ECG analysis.')

            files = request.files.getlist('files')

            if len(files) != 2:
                return render_template('index.html', message='Please upload exactly two files: one .hea and one .mat file.')

            file_hea, file_mat = None, None
            for file in files:
                if file.filename.endswith('.hea'):
                    file_hea = file
                elif file.filename.endswith('.mat'):
                    file_mat = file

            if file_hea is None or file_mat is None:
                return render_template('index.html', message='Please upload both .hea and .mat files.')

            header_contents = file_hea.read().decode('utf-8').splitlines()
            age, gender = None, None
            for line in header_contents:
                if line.startswith('#Age'):
                    age = int(line.strip().split(': ')[-1])
                if line.startswith('#Sex'):
                    gender_value = line.strip().split(': ')[-1]
                    gender = "Male" if gender_value == 'Male' else "Female"

            if age is None or gender is None:
                return render_template('index.html', message='Missing age or gender information in .hea file')

            mat_content = file_mat.read()
            mat = scipy.io.loadmat(BytesIO(mat_content))

            if 'val' in mat:
                numpy_array = mat['val']
                first_row = numpy_array[0]
                pre_processed_row = pre_processing(first_row, notch_frequency=60, notch_bandwidth=2, baseline_window_size=100, fs=500)

                num_samples = len(first_row)
                time = np.linspace(0, num_samples - 1, num_samples)
                plt.figure(figsize=(12, 6))
                plt.plot(time, pre_processed_row)
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude')
                plt.grid()

                img_bytes = BytesIO()
                plt.savefig(img_bytes, format='png')
                img_bytes.seek(0)
                plt.close()

                encoded_img = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

                features_dict = get_features(pre_processed_row, age)
                logging.debug(f"Features dictionary: {features_dict}")
                heart_rate = features_dict['HRM_Average_HR']

                X_new = flatten_features(features_dict).reshape(1, -1)
               
                logging.debug(f"Shape of X_new: {X_new.shape}")
                logging.debug(f"Flattened features: {X_new}")

                with open('HR_trained_model.pkl', 'rb') as f:
                    trained_models = pickle.load(f)

                y_pred_list = []
                for i, model in enumerate(trained_models):

                    y_pred_proba = model.predict_proba(X_new)
                    logging.debug(f"Model {i} predicted probabilities: {y_pred_proba}")
                    y_pred_list.append(y_pred_proba[0][1])

                logging.debug(f"Predicted probabilities list: {y_pred_list}")

                y_pred = np.array(y_pred_list)
                logging.debug(f"Prediction array: {y_pred}")

                HR_labels = ['Atrial Fibrillation', 'Sinus Rhythm', 'Sinus Bradycardia', 'Sinus Tachycardia']
                max_pred_index = np.argmax(y_pred)
                final_diagnosis = HR_labels[max_pred_index]
            else:
                return render_template('index.html', message='Invalid .mat file format')

            # Handle Temperature input
            temperature = None
            if 'temperature' in request.form:
                temperature = float(request.form['temperature'])
                temperature_diagnosis = get_diagnosis(temp_clf, temp_label_classes, np.array([[temperature]]))
                min_temp = 35.0  # Minimum temperature for color bar
                max_temp = 42.0  # Maximum temperature for color bar
                arrowPosition = (temperature - min_temp) / (max_temp - min_temp)
            else:
                temperature_diagnosis = 'Temperature input missing.'

            # Handle Blood Pressure input
            systolic, diastolic = None, None
            if 'bp_values' in request.form:
                bp_values = request.form['bp_values']
                if '/' in bp_values:
                    systolic, diastolic = map(float, bp_values.split('/'))
                    bp_diagnosis = get_diagnosis(bp_clf, bp_label_classes, np.array([[systolic, diastolic]]))
                else:
                    bp_diagnosis = 'Invalid input format for blood pressure. Please use "systolic/diastolic" format.'
            else:
                bp_diagnosis = 'Blood pressure input missing.'

            # Handle Respiration Rate input
            respiration_rate = None
            if 'respiration_rate' in request.form:
                respiration_rate = float(request.form['respiration_rate'])
                rr_diagnosis = get_diagnosis(rr_clf, rr_label_classes, np.array([[respiration_rate]]))
            else:
                rr_diagnosis = 'Respiration rate input missing.'

            # Handle SpO2 input
            spo2 = None
            if 'spo2' in request.form:
                spo2 = float(request.form['spo2'])
                spo2_diagnosis = get_diagnosis(spo2_clf, spo2_label_classes, np.array([[spo2]]))
            else:
                spo2_diagnosis = 'SpO2 input missing.'

            # Calculate health score and recommendations
            diagnoses = {
                'heart_rate': final_diagnosis,
                'blood_pressure': bp_diagnosis,
                'body_temperature': temperature_diagnosis,
                'respiratory_rate': rr_diagnosis,
                'spo2': spo2_diagnosis
            }

            health_score, recommendations = calculate_health_score(diagnoses)
            heart_rate_recommendation = recommendations['heart_rate']
            bp_recommendation = recommendations['blood_pressure']
            temperature_recommendation = recommendations['body_temperature']
            respiratory_rate_recommendation = recommendations['respiratory_rate']
            spo2_recommendation = recommendations['spo2']

            return render_template('result.html',
                                   message='Vitals analysis complete', 
                                   plot=encoded_img, 
                                   age=age, 
                                   gender=gender, 
                                   heart_rate=round(heart_rate),
                                   ecg_prediction=final_diagnosis, 
                                   temperature=temperature, 
                                   temperature_diagnosis=temperature_diagnosis,
                                   arrowPosition=arrowPosition,
                                   systolic=systolic,
                                   diastolic=diastolic,
                                   bp_diagnosis=bp_diagnosis,
                                   respiration_rate=respiration_rate,
                                   rr_diagnosis=rr_diagnosis,
                                   spo2=spo2,
                                   spo2_diagnosis=spo2_diagnosis,
                                   health_score=round(health_score),
                                   heart_rate_recommendation=heart_rate_recommendation,
                                   bp_recommendation=bp_recommendation,
                                   temperature_recommendation=temperature_recommendation,
                                   respiratory_rate_recommendation=respiratory_rate_recommendation,
                                   spo2_recommendation=spo2_recommendation)
        
        except Exception as e:
            logging.exception('An error occurred during processing')
            return render_template('index.html', message=f'An error occurred during processing: {str(e)}')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
