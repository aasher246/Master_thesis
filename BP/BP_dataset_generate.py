import numpy as np
import pandas as pd

# Define thresholds
ranges = {
    'hypotension': {'systolic': (90, 119), 'diastolic': (60, 79)},
    'normal': {'systolic': (120, 129), 'diastolic': (80, 84)},
    'elevated': {'systolic': (130, 139), 'diastolic': (85, 89)},
    'hypertension_stage_1': {'systolic': (140, 179), 'diastolic': (90, 119)},
    'hypertension_stage_2': {'systolic': (180, 219), 'diastolic': (120, 129)},
    'hypertensive_crisis': {'systolic': (220, 250), 'diastolic': (130, 150)}
}

# Function to generate random blood pressure values within specified range
def generate_random_blood_pressure(ranges):
    systolic = np.random.randint(ranges['systolic'][0], ranges['systolic'][1] + 1)
    diastolic = np.random.randint(ranges['diastolic'][0], ranges['diastolic'][1] + 1)
    return systolic, diastolic

# Generate data for each category
data = []
for category, range_values in ranges.items():
    for _ in range(1000):
        systolic, diastolic = generate_random_blood_pressure(range_values)
        data.append([systolic, diastolic, category])

# Create a DataFrame from the generated data
columns = ['Systolic', 'Diastolic', 'Category']
df = pd.DataFrame(data, columns=columns)

# Shuffle the DataFrame to randomize the order of samples
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the generated dataset to a CSV file
df.to_csv('blood_pressure_dataset.csv', index=False)
