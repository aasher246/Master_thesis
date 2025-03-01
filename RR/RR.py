import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Function to categorize respiration rates
def categorize_respiration_rate(rate):
    if rate < 12:
        return 'Bradypnoea'
    elif rate > 20:
        return 'Tachypnoea'
    else:
        return 'Normal'

# Load data
data_file = "respiration_rate_dataset.csv"
df = pd.read_csv(data_file)

# Categorize respiration rates
df['Category'] = df['RR(bpm)'].apply(categorize_respiration_rate)

# Prepare the data
X = df[['RR(bpm)']]  # Feature: Respiration rate
y = df['Category']   # Target variable: Respiration rate category

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define hyperparameter grid
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train_encoded)

# Get best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train classifier with best hyperparameters
best_clf = RandomForestClassifier(**best_params, random_state=42)
best_clf.fit(X_train, y_train_encoded)

# Evaluate the model
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Accuracy:", accuracy)
class_names = label_encoder.classes_
print("Classification Report:\n", classification_report(y_test_encoded, y_pred,target_names=class_names))

# Save the trained model and label encoder
joblib.dump(best_clf, 'RR_RandomForest_model.pkl')
np.save('rr_label_classes.npy', label_encoder.classes_)
