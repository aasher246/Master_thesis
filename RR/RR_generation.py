import numpy as np
import pandas as pd

# Define number of samples
num_samples = 1200  # Increased dataset size

# Generate synthetic respiratory rate data with some variation and imperfections
normal_rate = np.random.normal(16, 3, int(num_samples * 0.7)) + np.random.normal(0, 1, int(num_samples * 0.7))  # Adding noise
bradypnoea_rate = np.random.normal(8, 2, int(num_samples * 0.2)) + np.random.normal(0, 1, int(num_samples * 0.2))  # Adding noise
tachypnoea_rate = np.random.normal(24, 3, int(num_samples * 0.1)) + np.random.normal(0, 1, int(num_samples * 0.1))  # Adding noise

# Combine data from different classes and generate corresponding diagnoses
respiratory_rates = np.concatenate((normal_rate, bradypnoea_rate, tachypnoea_rate))
diagnoses = np.concatenate((['Normal'] * len(normal_rate), ['Bradypnoea'] * len(bradypnoea_rate), ['Tachypnoea'] * len(tachypnoea_rate)))

# Shuffle the data
indices = np.arange(len(respiratory_rates))
np.random.shuffle(indices)
respiratory_rates = respiratory_rates[indices]
diagnoses = diagnoses[indices]

# Round respiratory rates to whole numbers
respiratory_rates = np.round(respiratory_rates).astype(int)

# Ensure respiratory rates are within realistic bounds
respiratory_rates = np.clip(respiratory_rates, 5, 30)

# Create a DataFrame
df = pd.DataFrame({'RR(bpm)': respiratory_rates, 'Diagnosis': diagnoses})

# Save the DataFrame to a CSV file
df.to_csv('respiration_rate_dataset.csv', index=False)

