import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import savgol_filter
import Functions_HR
from Functions_HR import pre_processing



################################   EXTRACTING THE FILES TO USE   ################################

data_path = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords/01/ALL1'
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

diagnoses = []

disease_codes_to_exclude = ['55827005','251146004','47665007','164912004','47665007','233917008','251199005','164931005','59931005', '195042002', '54016002', '28189009', '27885002', '251173003', '251198002', '428417006', '164942001', '426995002', '251164006', '164873001', '251148003', '251147008', '164865005', '164947007', '111975006', '446358003', '89792004', '164930006', '164937009', '11157007', '75532003', '13640000', '195060002', '251180001', '195101003', '74390002', '427393009', '426761007', '713422000', '233896004', '233897008', '195101003']


# Loop through each .hea file
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
                    # Extract the first diagnosis
                    first_diagnosis = diagnosis.split(',')[0]
                    hea_files_to_keep.append(file)
                    hea_single_diagnosis.append(first_diagnosis)  # Append to the list of files with single diagnosis
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

patients_diagnosis = [get_diagnosis_names(diagnosis) for diagnosis in diagnoses]
patients_with_single_diagnosis = [get_diagnosis_names(diagnosis) for diagnosis in hea_single_diagnosis]

 
##################################   LOADING MAT DATA #######################################
csv_path = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/CSV'
txt_path = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/TXT'
for i,file in enumerate(hea_files_to_keep):
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
    ecg_array=ecg_array[0]
    
    num_samples = len(ecg_array)
    time = np.linspace(0, num_samples - 1, num_samples)  # Create time axis
    
    ecg_array_filtered = pre_processing(ecg_array, notch_frequency=60, notch_bandwidth=2, baseline_window_size=100, fs=500 )
    ecg_final.append(ecg_array_filtered)
  
    #num_leads, num_samples = ecg_array.shape
    
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2,1,1)
    # plt.title(f'ECG Signal with Low-Pass Filter - {hea_files_to_keep[i]} Diagnosis: {patients_with_single_diagnosis[i]}')
    # plt.plot(time, ecg_array, label='Original ECG')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    
    
    # plt.subplot(2,1,2)
    # plt.plot(time, ecg_array_filtered, label='Filtered ECG')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()


# import os
# import pandas as pd

# # Creating a DataFrame to store patient names and diagnoses
# data = {'Patient Name': root, 'Diagnosis': patients_with_single_diagnosis}
# df = pd.DataFrame(data)

# # Saving the DataFrame to a CSV file
# csv_output_path = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/Patients&DiagnosisTest.csv'  # Specify the desired output path
# df.to_csv(csv_output_path, index=False)

# print("CSV file with patient names and diagnoses created successfully.")
