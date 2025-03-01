import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import updatedstuff

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

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)),
    Dropout(0.5),  # Add dropout layer for regularization
    Dense(32, activation='relu'),
    Dense(y_train_resampled.shape[1], activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with epochs
model.fit(X_train_resampled, y_train_resampled, epochs=200, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Filter out classes with 0 support
non_zero_classes = [class_label for class_label, support in zip(range(len(y_test[0])), y_test.sum(axis=0)) if support > 0]
filtered_y_test = y_test[:, non_zero_classes]
filtered_y_pred = y_pred[:, non_zero_classes]

diagnosis_list= list(diagnosis_list)

# Print classification report
print(classification_report(filtered_y_test, filtered_y_pred, target_names=[diagnosis_list[i] for i in non_zero_classes]))

# Serialize the trained model to a file
model.save('trained_neural_network_model.h5')
