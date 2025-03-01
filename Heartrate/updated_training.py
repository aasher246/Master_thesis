from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import updatedstuff
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE

# Extract features and labels from all_dict
features = []
labels = []

all_dict = updatedstuff.all_dict
diagnosis_list = updatedstuff.unique_diagnoses

# Extract features and binary labels for each disease
for patient_data in all_dict:
    feature_vector = [
        patient_data['Age'],
        patient_data['EWA_ABSAmp_P'],
        patient_data['EWA_ABSAmp_Q'],
        patient_data['EWA_ABSAmp_R'],
        patient_data['EWA_ABSAmp_S'],
        patient_data['EWA_ABSAmp_T'],
        patient_data['EWA_Amp_R_to_P'],
        patient_data['EWA_Amp_R_to_T'],
        patient_data['FA_Frequency_Width'],
        patient_data['FA_Peak_Frequency'],
        patient_data['FA_Spectral_Entropy'],
        patient_data['FA_Total_Power'],
        patient_data['HRM_Average_HR'],
        patient_data['HRM_Max_HR'],
        patient_data['HRM_Min_HR'],
        patient_data['HRM_PNN50'],
        patient_data['HRM_RMSSD'],
        patient_data['HRM_Sdnn_RR'],
        patient_data['Mean_P'],
        patient_data['Mean_Q'],
        patient_data['Mean_R'],
        patient_data['Mean_S'],
        patient_data['Mean_T'],
        
        patient_data['Median_P'],
        patient_data['Median_Q'],
        patient_data['Median_R'],
        patient_data['Median_S'],
        patient_data['Median_T'],
        
        patient_data['cv_P'],
        patient_data['cv_Q'],
        patient_data['cv_R'],
        patient_data['cv_S'],
        patient_data['cv_T'],
        
        patient_data['Std_Dev_P'],
        patient_data['Std_Dev_Q'],
        patient_data['Std_Dev_R'],
        patient_data['Std_Dev_S'],
        patient_data['Std_Dev_T'],
        
        patient_data['Range_P'],
        patient_data['Range_Q'],
        patient_data['Range_R'],
        patient_data['Range_S'],
        patient_data['Range_T'],

        
        # Add other features as needed
    ]
    features.append(feature_vector)
    binary_label = np.zeros(len(diagnosis_list))
    for i, diagnosis in enumerate(diagnosis_list):
        if diagnosis in patient_data['Diagnosis']:
            binary_label[i] = 1
    labels.append(binary_label)

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE oversampling to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train a separate classifier for each label
classifiers = []
best_params = []
for i in range(y_train_resampled.shape[1]):
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=StratifiedKFold(3), scoring='accuracy')
    grid_search.fit(X_train_resampled, y_train_resampled[:, i])
    classifiers.append(grid_search.best_estimator_)
    best_params.append(grid_search.best_params_)

# Predict the probabilities for each label
y_proba = np.hstack([clf.predict_proba(X_test)[:, 1].reshape(-1, 1) for clf in classifiers])

# Make predictions on the test set using the best model
y_pred = (y_proba > 0.5).astype(int)

# Filter out classes with 0 support
non_zero_classes = [class_label for class_label, support in zip(range(len(y_test[0])), y_test.sum(axis=0)) if support > 0]
filtered_y_test = y_test[:, non_zero_classes]
filtered_y_pred = y_pred[:, non_zero_classes]

diagnosis_list= list(diagnosis_list)

# Evaluate the model
accuracy = accuracy_score(filtered_y_test, filtered_y_pred)
print("Accuracy:", accuracy)
print("Best Parameters for Each Label:")
for i, params in enumerate(best_params):
    print(f"Label {i + 1}: {params}")
print("Classification Report:")
print(classification_report(filtered_y_test, filtered_y_pred, target_names=[diagnosis_list[i] for i in non_zero_classes], zero_division=1))

# Serialize the trained model to a file
with open('HR_trained_modelz.pkl', 'wb') as f:
    pickle.dump(classifiers, f)