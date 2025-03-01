# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import joblib

# # Load the generated temperature dataset
# df = pd.read_csv("temperature_dataset.csv")

# # Extract temperature values and categories
# X = df[['Temperature']]
# y = df['Category']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Encode categorical labels into numerical representations
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)

# # Calculate class weights
# # class_weights = {label: 1 / count for label, count in zip(np.unique(y_train_encoded), np.bincount(y_train_encoded))}

# # Create and train the logistic regression classifier
# clf = LogisticRegression(class_weight='balanced', random_state=42)
# clf.fit(X_train, y_train_encoded)

# # Make predictions on the test set
# y_pred = clf.predict(X_test)

# # Decode predictions back to original labels if needed
# y_pred_decoded = label_encoder.inverse_transform(y_pred)

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
# class_names = label_encoder.classes_
# print("Classification Report:\n", classification_report(y_test_encoded, y_pred, target_names=class_names, zero_division=1))

# # Save the trained model and label encoder classes
# joblib.dump(clf, 'temp_logistic_regression_model.pkl')
# np.save('temp_label_classes.npy', label_encoder.classes_)


# train_temperature_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the generated temperature dataset
df = pd.read_csv("temperature_dataset.csv")

# Extract temperature values and categories
X = df[['Temperature']]
y = df['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode categorical labels into numerical representations
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create and train the RandomForestClassifier
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train_encoded)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Decode predictions back to original labels
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
class_names = label_encoder.classes_
print("Classification Report:\n", classification_report(y_test_encoded, y_pred, target_names=class_names, zero_division=1))

# Save the trained model and label encoder classes
joblib.dump(clf, 'temp_random_forest_model.pkl')
np.save('temp_label_classes.npy', label_encoder.classes_)

#temp_label_classes = np.load('temp_label_classes.npy', allow_pickle=True)
