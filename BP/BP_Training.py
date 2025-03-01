import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the generated blood pressure dataset
df = pd.read_csv("blood_pressure_dataset.csv")

# Extract blood pressure values and categories
X = df[['Systolic', 'Diastolic']]
y = df['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode categorical labels into numerical representations
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Calculate class weights
class_weights = {label: 1 / count for label, count in zip(np.unique(y_train_encoded), np.bincount(y_train_encoded))}

# Create and train the Random Forest classifier with class weights
clf = RandomForestClassifier(class_weight=class_weights, random_state=42)
clf.fit(X_train, y_train_encoded)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Decode predictions back to original labels if needed
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
# print("Classification Report:\n", classification_report(y_test_encoded, y_pred, zero_division=1))

# Make predictions on the independent test set
X_test_independent = df[['Systolic', 'Diastolic']]
y_test_independent = df['Category']
y_pred_independent = clf.predict(X_test_independent)
y_pred_decoded_independent = label_encoder.inverse_transform(y_pred_independent)

# Evaluate the model on the independent test set
accuracy_independent = accuracy_score(y_test_independent, y_pred_decoded_independent)
classification_report_independent = classification_report(y_test_independent, y_pred_decoded_independent)

print("Accuracy on independent test set:", accuracy_independent)
print("Classification Report on independent test set:\n", classification_report_independent)

# Example input data (systolic and diastolic blood pressure values)
input_data = [[120, 80], [140, 90], [150, 120]]  # Example input: [systolic, diastolic]

# Make predictions using the model
predictions = clf.predict(input_data)

# Display the predictions
print("Predictions:")
for i, pred in enumerate(predictions):
    systolic, diastolic = input_data[i]
    print(f"Input {i+1}: Systolic - {systolic}, Diastolic - {diastolic} - Predicted class: {label_encoder.inverse_transform([pred])[0]}")

# Save the trained model
joblib.dump(clf, 'blood_pressure_model.pkl')
np.save('bp_label_classes.npy', label_encoder.classes_)
