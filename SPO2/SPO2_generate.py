import numpy as np
import pandas as pd

# Define thresholds with one decimal place
ranges = {
    'Normal Blood Oxygen levels': {'spo2': (95.0, 100.0)},
    'Concerning Blood Oxygen levels': {'spo2': (91.0, 95.0)},
    'Low Blood Oxygen levels': {'spo2': (90.0, 91.0)},
    'Low Blood Oxygen levels that can affect your brain': {'spo2': (80.0, 85.0)},
    'Cyanosis': {'spo2': (67.0, 67.0)}
}

# Function to generate random SpO2 values within specified range with one decimal place
def generate_random_spo2(ranges):
    spo2 = round(np.random.uniform(ranges['spo2'][0], ranges['spo2'][1]), 1)
    return spo2

# Generate data for each category
data = []
for category, range_values in ranges.items():
    num_samples = np.random.randint(100, 300)  # Generate a random number of samples for each category
    for _ in range(num_samples):
        spo2 = generate_random_spo2(range_values)
        data.append([spo2, category])

# Create a DataFrame from the generated data
columns = ['SpO2', 'Category']
df = pd.DataFrame(data, columns=columns)

# Shuffle the DataFrame to randomize the order of samples
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the generated dataset to a CSV file
df.to_csv('spo2_dataset.csv', index=False)
