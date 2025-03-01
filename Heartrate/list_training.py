from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
import updatedstuff
from keras.preprocessing.sequence import pad_sequences  # Import pad_sequences from keras

all_list_dict = updatedstuff.all_list_dict
diagnosis_list = updatedstuff.unique_diagnoses

features_list = []
labels_list = []

for patient_data in all_list_dict:
    feature_list_vector = [
        patient_data['EWA_P_Int'],
        patient_data['EWA_QRS_Int'],
        patient_data['EWA_T_Int'],
        patient_data['EWA_Amp_P'],
        patient_data['EWA_Amp_Q'],
        patient_data['EWA_Amp_R'],
        patient_data['EWA_Amp_S'],
        patient_data['EWA_Amp_T']
        # Add other features as needed
    ]
    features_list.append(feature_list_vector)
    binary_list_label = np.zeros(len(diagnosis_list))
    for i, diagnosis in enumerate(diagnosis_list):
        if diagnosis in patient_data['Diagnosis']:
            binary_list_label[i] = 1
    labels_list.append(binary_list_label)

# Find the maximum sequence length
max_sequence_length = max(len(seq) for seq in features_list)

# Pad each sequence to the maximum length individually
features_list_padded = []
for seq in features_list:
    padded_seq = pad_sequences([seq], maxlen=max_sequence_length, padding='post', dtype='float32')
    features_list_padded.append(padded_seq)

# Convert to numpy array
X_lists = np.array(features_list_padded)
y_lists = np.array(labels_list)

# Split the data into training and testing sets
X_train_lists, X_test_lists, y_train_lists, y_test_lists = train_test_split(X_lists, y_lists, test_size=0.2, random_state=42)

# Train separate classifier for each label
classifiers_lists = []
best_params_lists = []
for i, label in enumerate(diagnosis_list):
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    clf.fit(X_train_lists[:, 0], y_train_lists[:, i])  # Use only the first sequence for training
    classifiers_lists.append(clf)

# Make predictions
y_proba_lists = np.hstack([clf.predict_proba(X_test_lists[:, 0])[:, 1].reshape(-1, 1) for clf in classifiers_lists])  # Use only the first sequence for prediction

# Combine predictions
y_pred_lists = (y_proba_lists > 0.7).astype(int)

# Evaluate the model
accuracy_lists = accuracy_score(y_test_lists, y_pred_lists)
print("Accuracy:", accuracy_lists)
print("Classification Report:")
print(classification_report(y_test_lists, y_pred_lists, target_names=diagnosis_list, zero_division=1))

# Serialize the trained model to a file
with open('HR_trained_model_lists.pkl', 'wb') as f:
    pickle.dump(classifiers_lists, f)
