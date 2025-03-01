import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import savgol_filter
import csv
import neurokit2 as nk



################################   EXTRACTING THE FILES TO USE   ################################

data_path = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords/01/ALL'
hea_files = [f for f in os.listdir(data_path) if f.endswith('.hea')]  #list of all .hea files
# Loading the CSV file with the mapping of Snomed_CT codes to condition names
csv_file_path = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/ConditionNames_SNOMED-CT.csv'
mapping_df = pd.read_csv(csv_file_path)

ecg_final=[]
root=[]
age,sex=[],[]

#creating a new list to store the names of .hea files without the unwanted diagnosis
hea_files_to_keep = []
hea_single_diagnosis=[]
# Disease code to exclude
disease_code_to_exclude = '55827005'

diagnoses = []
for file in hea_files:
    record_name = file.split('.')[0]
    record_path = os.path.join(data_path, record_name)
    
    with open(record_path + '.hea', 'r') as f:
        header_contents = f.readlines()
        
        # Finding the 'Dx' parameter
        for line in header_contents:
            if line.startswith('#Dx'):
                diagnosis = line.strip().split(': ')[-1]
                if disease_code_to_exclude not in diagnosis:
                    hea_files_to_keep.append(file)
                break


disease_codes_to_exclude = ['55827005', '195042002', '54016002', '28189009', '27885002', '251173003', '251198002', '428417006', '164942001', '426995002', '251164006', '164873001', '251148003', '251147008', '164865005', '164947007', '111975006', '446358003', '89792004', '164930006', '164937009', '11157007', '75532003', '13640000', '195060002', '251180001', '195101003', '74390002', '427393009', '426761007', '713422000', '233896004', '233897008', '195101003']

hea_files_to_keep = []

for file in hea_files:
    record_name = file.split('.')[0]
    record_path = os.path.join(data_path, record_name)
    
    with open(record_path + '.hea', 'r') as f:
        header_contents = f.readlines()
        
        # Finding the 'Dx' parameter
        for line in header_contents:
            if line.startswith('#Dx'):
                diagnosis = line.strip().split(': ')[-1]
                
                # Check if any disease code matches the exclusion codes
                if not any(code in diagnosis for code in disease_codes_to_exclude):
                    hea_files_to_keep.append(file)
                break


################################  MAPPING DIAGNOSIS NAMES   ################################

# Dictionary to map Snomed_CT codes to condition names
diagnosis_mapping = dict(zip(mapping_df['Snomed_CT'], mapping_df['Full Name']))

#### FUNCTION START ###
# Function to handle single and multiple diagnoses
def get_diagnosis_names(diagnosis_codes):
    
    if isinstance(diagnosis_codes, list):
        return [diagnosis_mapping.get(int(code), 'Unknown') for code in diagnosis_codes]
    elif isinstance(diagnosis_codes, str):
        codes_list = diagnosis_codes.split(',')
        return [diagnosis_mapping.get(int(code.strip()), 'Unknown') for code in codes_list]
    else:
        return 'Unknown'
    
    
for file in hea_files_to_keep:
    record_name = file.split('.')[0]
    record_path = os.path.join(data_path, record_name)
    
    with open(record_path + '.hea', 'r') as f:
        header_contents = f.readlines()
        
        # Find the line that contains the 'Dx' parameter
        for line in header_contents:
            if line.startswith('#Dx'):
                diagnosis = line.strip().split(': ')[-1]
                diagnoses.append(diagnosis)
                diagnosis = diagnosis.split(',')
                if len(diagnosis) > 0:
                    hea_single_diagnosis.append(file)
                    record,ext= file.split('.')
                    root.append(record)
                    break
            if line.startswith('#Age'):
                a= line.strip().split(': ')[-1]
                age.append(a)
            if line.startswith('#Sex'):
                s= line.strip().split(': ')[-1]
                sex.append(s)
#### FUNCTION END ###

patients_with_single_diagnosis = [get_diagnosis_names(diagnosis) for diagnosis in diagnoses]
#patients_with_single_diagnosis = [diagnosis_list for diagnosis_list in diagnoses_with_names if len(diagnosis_list) == 1]

######################################### DENOISING #########################################

#### FUNCTION START ###
# Function for Denoising the Signal (LPF, Notch Filter, baseline wander correction, and smoothing)
    
def apply_signal_processing(ecg_array, notch_frequency, notch_bandwidth, baseline_window_size, fs=500):
    num_leads, num_samples = ecg_array.shape
    
    # Designing a notch filter to remove powerline interference (50-60Hz)
    nyquist_frequency = 0.5 * fs
    notch_freq = notch_frequency / nyquist_frequency
    notch_bw = notch_bandwidth / nyquist_frequency
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, notch_bw)

    # Applying the notch filter to each lead of the ECG array
    ecg_array_filtered = np.zeros_like(ecg_array)
    for lead in range(num_leads):
        # Apply notch filter
        ecg_array_filtered[lead] = scipy.signal.lfilter(b_notch, a_notch, ecg_array[lead])
    
    # Baseline wander correction using moving average
    corrected_ecg_array = np.zeros_like(ecg_array_filtered)
    for lead in range(num_leads):
        moving_avg = np.convolve(ecg_array_filtered[lead], np.ones(baseline_window_size)/baseline_window_size, mode='same')
        corrected_ecg_array[lead] = ecg_array_filtered[lead] - moving_avg
    
    # Apply smoothing filter to each lead of the corrected ECG array
    smoothed_ecg_array = np.zeros_like(corrected_ecg_array)
    # Smoothing parameters
    window_size = 25  # Should be an odd number
    order = 2  # Polynomial order
    for lead in range(num_leads):
        smoothed_ecg_array[lead] = savgol_filter(corrected_ecg_array[lead], window_size, order)
    return smoothed_ecg_array

# #NORMALIZING THE VALUES
# def normalize_ecg_array(ecg_array):
#     min_value = np.min(ecg_array)
#     max_value = np.max(ecg_array)

#     ecg_normalized_array = (ecg_array - min_value) / (max_value - min_value)
#     return ecg_normalized_array

# ### FUNCTION END ###

def normalize_ecg_array(ecg_array):
    # Calculate percentile values for clipping
    clip_min = np.percentile(ecg_array, 1)
    clip_max = np.percentile(ecg_array, 99)

    # Clip values to the calculated percentiles
    ecg_array = np.clip(ecg_array, clip_min, clip_max)

    # Normalize the clipped array
    min_value = np.min(ecg_array)
    max_value = np.max(ecg_array)
    ecg_normalized_array = (ecg_array - min_value) / (max_value - min_value)
    return ecg_normalized_array

 
##################################   LOADING MAT DATA #######################################
csv_path = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/CSV'
txt_path = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/TXT'
for i,file in enumerate(hea_single_diagnosis):
    record_name = file.split('.')[0]
    record_path = os.path.join(data_path, record_name)
    
    with open(record_path + '.hea', 'r') as f:
        header_contents = f.readlines()
        
        # Finding the line that contains the 'Dx' parameter
        for line in header_contents:
            if line.startswith('#Dx'):
                diagnosis = line.strip().split(': ')[-1]
                break
    
    # Load ECG data from the .mat file
    mat_data = scipy.io.loadmat(record_path + '.mat')
    ecg_array = mat_data['val']  # This will contain the ECG signal
    
    num_samples = len(ecg_array[0])
    time = np.linspace(0, num_samples - 1, num_samples)  # Create time axis
    ecg_array_filtered = apply_signal_processing(ecg_array, notch_frequency=60, notch_bandwidth=2, baseline_window_size=100, )
    ecg_final.append(ecg_array_filtered)
    
    #NORMALIZING THE ECG VALUES
    ecg_normalized = normalize_ecg_array(ecg_array)
    ecg_normalized_filtered = normalize_ecg_array(ecg_array_filtered)
    
    
    
    time = np.arange(ecg_array.shape[1]) / 500
  
num_leads, num_samples = ecg_array.shape


for lead in range(5):
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.title(f'ECG Signal with Low-Pass Filter - {hea_single_diagnosis[i]} Diagnosis: {patients_with_single_diagnosis[i]}')
    plt.plot(time, ecg_array[lead], label='Original ECG')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(time, ecg_array_filtered[lead], label='Filtered ECG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()


    # plt.figure(figsize=(12, 6))
    # plt.subplot(2,1,1)
    # plt.title(f'ECG Signal with Low-Pass Filter - {hea_single_diagnosis[i]} Diagnosis: {patients_with_single_diagnosis[i]}')
    # plt.plot(time, ecg_array[0], label='Original ECG')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    
    # plt.subplot(2,1,2)
    # plt.plot(time, ecg_array_filtered[0], label='Filtered ECG')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()