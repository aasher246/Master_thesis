from keras.layers import Input, Dense, Add
from keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import Feature_extraction_single_input_Heartrate


# Your existing code ...

num_samples = 5000
diagnosis_class_count = Feature_extraction_single_input_Heartrate.unique_diagnoses_count
diagnosis = Feature_extraction_single_input_Heartrate.diagnosis
patients = Feature_extraction_single_input_Heartrate.all_dict
ecg_signals = Feature_extraction_single_input_Heartrate.all_ecg
all_gender = Feature_extraction_single_input_Heartrate.sex
num_features = 15

# Initialize a tokenizer for diagnoses
tokenizer = Tokenizer()

# Lists to store patient inputs and labels
patient_inputs = []
encoded_diagnoses = []

# Lists to store sequences for padding
amplitudes_p_sequences = []
amplitudes_q_sequences = []
amplitudes_r_sequences = []
amplitudes_s_sequences = []
amplitudes_t_sequences = []
pr_intervals_sequences = []
pr_segments_sequences = []
qrs_intervals_sequences = []
qt_intervals_sequences = []
rr_intervals_sequences = []
st_intervals_sequences = []
st_segments_sequences = []
heartrate_sequences = []
age_sequences = []
gender_sequences = []

gender_mapping = {"Male": 0, "Female": 1}
encoded_gender = [gender_mapping[gender] for gender in all_gender]

for patient_data, gen in zip(patients, encoded_gender):
    amplitudes_p = np.array(patient_data["Amplitudes_P"])
    amplitudes_q = np.array(patient_data["Amplitudes_Q"])
    amplitudes_r = np.array(patient_data["Amplitudes_R"])
    amplitudes_s = np.array(patient_data["Amplitudes_S"])
    amplitudes_t = np.array(patient_data["Amplitudes_T"])
    pr_intervals = np.array(patient_data["PR_Intervals"])
    pr_segments = np.array(patient_data["PR_Segments"])
    qrs_intervals = np.array(patient_data["QRS_Intervals"])
    qt_intervals = np.array(patient_data["QT_Intervals"])
    rr_intervals = np.array(patient_data["RR_Intervals"])
    st_intervals = np.array(patient_data["ST_Intervals"])
    st_segments = np.array(patient_data["ST_Segments"])
    age = np.array([int(patient_data["Age"])])
    heartrate = np.array(patient_data["Heart_Rate"])

    # # Pad the sequences to the maximum length
    # amplitudes_p_sequence = pad_sequences([amplitudes_p], maxlen=max_sequence_length, dtype='float32', padding='post')
    # amplitudes_q_sequence = pad_sequences([amplitudes_q], maxlen=max_sequence_length, dtype='float32', padding='post')
    # amplitudes_r_sequence = pad_sequences([amplitudes_r], maxlen=max_sequence_length, dtype='float32', padding='post')
    # amplitudes_s_sequence = pad_sequences([amplitudes_s], maxlen=max_sequence_length, dtype='float32', padding='post')
    # amplitudes_t_sequence = pad_sequences([amplitudes_t], maxlen=max_sequence_length, dtype='float32', padding='post')
    # pr_intervals_sequence = pad_sequences([pr_intervals], maxlen=max_sequence_length, dtype='float32', padding='post')
    # pr_segments_sequence = pad_sequences([pr_segments], maxlen=max_sequence_length, dtype='float32', padding='post')
    # qrs_intervals_sequence = pad_sequences([qrs_intervals], maxlen=max_sequence_length, dtype='float32', padding='post')
    # qt_intervals_sequence = pad_sequences([qt_intervals], maxlen=max_sequence_length, dtype='float32', padding='post')
    # rr_intervals_sequence = pad_sequences([rr_intervals], maxlen=max_sequence_length, dtype='float32', padding='post')
    # st_intervals_sequence = pad_sequences([st_intervals], maxlen=max_sequence_length, dtype='float32', padding='post')
    # st_segments_sequence = pad_sequences([st_segments], maxlen=max_sequence_length, dtype='float32', padding='post')

    # Append the padded sequences to the respective lists
    amplitudes_p_sequences.append(amplitudes_p)
    amplitudes_q_sequences.append(amplitudes_q)
    amplitudes_r_sequences.append(amplitudes_r)
    amplitudes_s_sequences.append(amplitudes_s)
    amplitudes_t_sequences.append(amplitudes_t)
    pr_intervals_sequences.append(pr_intervals)
    pr_segments_sequences.append(pr_segments)
    qrs_intervals_sequences.append(qrs_intervals)
    qt_intervals_sequences.append(qt_intervals)
    rr_intervals_sequences.append(rr_intervals)
    st_intervals_sequences.append(st_intervals)
    st_segments_sequences.append(st_segments)
    age_sequences.append(age)
    heartrate_sequences.append(heartrate)

#Convert the lists of padded sequences to arrays
amplitudes_p_in = np.array(amplitudes_p_sequences)
amplitudes_q_in = np.array(amplitudes_q_sequences)
amplitudes_r_in = np.array(amplitudes_r_sequences)
amplitudes_s_in = np.array(amplitudes_s_sequences)
amplitudes_t_in = np.array(amplitudes_t_sequences)
pr_intervals_in = np.array(pr_intervals_sequences)
pr_segments_in = np.array(pr_segments_sequences)
qrs_intervals_in = np.array(qrs_intervals_sequences)
qt_intervals_in = np.array(qt_intervals_sequences)
rr_intervals_in = np.array(rr_intervals_sequences)
st_intervals_in = np.array(st_intervals_sequences)
st_segments_in = np.array(st_segments_sequences)

# Add an extra dimension to the last three features
heartrate_in = np.array(heartrate_sequences)
age_in = np.array(age_sequences)
gender_in = np.array(encoded_gender)

# Append inputs and labels to their respective lists
patient_inputs = [amplitudes_p_in, amplitudes_q_in, amplitudes_r_in, amplitudes_s_in, amplitudes_t_in, pr_intervals_in, pr_segments_in, qrs_intervals_in, qt_intervals_in, rr_intervals_in, st_intervals_in, st_segments_in, heartrate_in, age_in, gender_in]

# Tokenize the diagnosis labels
tokenizer = Tokenizer(oov_token=None)
tokenizer.fit_on_texts(diagnosis)
encoded_diagnoses = tokenizer.texts_to_matrix(diagnosis, mode='binary')[:, 1:]
diagnosis_class_count = len(tokenizer.word_index)  # Number of unique diagnoses

# Split each feature into training and testing sets
train_split = int(0.8 * len(patient_inputs[0]))  # Assuming 82 is the number of rows in your data
X_train = []
X_test = []

for feature_array in patient_inputs:
    # Slice the array to get the training and testing sets
    feature_train = feature_array[:train_split]
    feature_test = feature_array[train_split:]
    
    X_train.append(feature_train)
    X_test.append(feature_test)
    
    
y_train = encoded_diagnoses[:train_split]
y_test = encoded_diagnoses[train_split:]

# Apply SMOTE to oversample the minority class in training data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model architecture (as previously defined)
num_samples = 5000
num_channels = 1

# Define input shapes
amplitudes_p_input = Input(shape=(1,))
amplitudes_q_input = Input(shape=(1,))
amplitudes_r_input = Input(shape=(1,))
amplitudes_s_input = Input(shape=(1,))
amplitudes_t_input = Input(shape=(1,))
pr_intervals_input = Input(shape=(1,))
pr_segments_input = Input(shape=(1,))
qrs_intervals_input = Input(shape=(1,))
qt_intervals_input = Input(shape=(1,))
rr_intervals_input = Input(shape=(1,))
st_intervals_input = Input(shape=(1,))
st_segments_input = Input(shape=(1,))
heartrate_input = Input(shape=(1,))
age_input = Input(shape=(1,))
sex_input = Input(shape=(1,))

# Combine the features
amplitudes_input = [amplitudes_p_input, amplitudes_q_input, amplitudes_r_input, amplitudes_s_input, amplitudes_t_input, pr_intervals_input, pr_segments_input, qrs_intervals_input, qt_intervals_input, rr_intervals_input, st_intervals_input, st_segments_input, heartrate_input, age_input, sex_input]

amplitudes_features = Add()(amplitudes_input)
amplitudes_features_subnetwork = Dense(64, activation='relu')(amplitudes_features)
amplitudes_features_subnetwork = Dense(128, activation='relu')(amplitudes_features_subnetwork)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Additional processing layers
combined = Dense(32, activation='relu')(amplitudes_features_subnetwork)

# Output layer for multi-label diagnosis prediction
diagnosis_output = Dense(diagnosis_class_count, activation='sigmoid')(combined)

batch_size = 32  # You can set the batch size as needed
num_epochs = 100

# Create the model with multi-label diagnosis output
model = Model(inputs=amplitudes_input, outputs=diagnosis_output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.fit(X_train_resampled, y_train_resampled, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping])


# Assuming a threshold of 0.5 for binary prediction
threshold = 0.3  # Adjust this value
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='micro')
recall = recall_score(y_test, y_pred_classes, average='micro')

# Display metrics
# print(f'Accuracy: {accuracy:.4f}')
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')

# # Generate and display the confusion matrix
# conf_matrix = multilabel_confusion_matrix(y_test, y_pred_classes)
# for i, matrix in enumerate(conf_matrix):
#     print(f'\nConfusion Matrix for class {i}:\n{matrix}')