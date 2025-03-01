import numpy as np
import pandas as pd

# Define thresholds with one decimal place
ranges = {
    'Low temperature': {'temp': (30.0, 35.8)},
    'Normal': {'temp': (35.9, 37.0)},
    'Elevated temperature': {'temp': (37.1, 38.0)},
    'Moderate fever': {'temp': (38.1, 38.5)},
    'High fever': {'temp': (38.6, 39.5)},
    'Very high fever': {'temp': (39.6, 42.0)}
}

# Function to generate random temperature values within specified range with one decimal place
def generate_random_temperature(ranges):
    temp = round(np.random.uniform(ranges['temp'][0], ranges['temp'][1]), 1)
    return temp

# Generate data for each category
data = []
for category, range_values in ranges.items():
    for _ in range(500):
        temp = generate_random_temperature(range_values)
        data.append([temp, category])

# Create a DataFrame from the generated data
columns = ['Temperature', 'Category']
df = pd.DataFrame(data, columns=columns)

# Shuffle the DataFrame to randomize the order of samples
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the generated dataset to a CSV file
df.to_csv('temperature_dataset.csv', index=False)

