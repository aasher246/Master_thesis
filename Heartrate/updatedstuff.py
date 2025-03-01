# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import Heartrate_Initial
import Functions_HR
import math
import os
import statsmodels.api as sm
from scipy.signal import welch
from scipy.signal import spectrogram
from Functions_HR import get_R_amps, get_amps, get_peaks, get_interval, custom_function, fill_peaks, fill_amps, calculate_statistics
import matplotlib.pyplot as plt
import nolds
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
import warnings
import shutil

# Define the source and destination directories
source_dir = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords/01/ALL1'
destination_dir = '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords/01/Bad_files'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Custom warning handler
class SkipFileWarning(Warning):
    pass

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if "Too few peaks detected to compute the rate" in str(message):
        raise SkipFileWarning

warnings.showwarning = custom_warning_handler

all_dict = []
all_list_dict= []
all_ecg = []
all_gender = []

# Retrieve ECG data
list_of_patient_data = Heartrate_Initial.ecg_final
names = Heartrate_Initial.root
diagnosis = Heartrate_Initial.patients_with_single_diagnosis

flat_diagnoses = [diag for sublist in diagnosis for diag in sublist]
unique_diagnoses = set(flat_diagnoses)
unique_diagnoses_count = len(unique_diagnoses)

sex = Heartrate_Initial.sex
age = Heartrate_Initial.age

for x, patient_data in enumerate(list_of_patient_data):
    try:
        ecg_signal = patient_data
        fs = 500
        patient = names[x]
        num_samples = ecg_signal.shape

        time = len(ecg_signal) / fs
        peaks_R, peaks_P, peaks_Q, peaks_S, peaks_T, onsets_P, onsets_R, onsets_T, offsets_P, offsets_R, offsets_T, seg_PR, seg_ST, int_PR, int_ST, int_QT, int_RR, int_QRS, amp_R, amp_P, amp_Q, amp_S, amp_T = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        # Extracting R-peaks locations
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
        R = rpeaks['ECG_R_Peaks'].tolist()
        R = [j * 2 for j in R]

        # Delineating the ECG signal for peaks
        _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak")

        # Getting the wave boundaries
        signal_cwt, waves_cwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="cwt")
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="dwt")

        peaks_P = get_peaks(waves_dwt['ECG_P_Peaks'])
        peaks_Q = get_peaks(waves_dwt['ECG_Q_Peaks'])
        peaks_R = get_peaks(rpeaks['ECG_R_Peaks'])
        peaks_S = get_peaks(waves_dwt['ECG_S_Peaks'])
        peaks_T = get_peaks(waves_dwt['ECG_T_Peaks'])

        onsets_P = get_peaks(waves_dwt['ECG_P_Onsets'])
        onsets_R = get_peaks(waves_dwt['ECG_R_Onsets'])
        onsets_T = get_peaks(waves_dwt['ECG_T_Onsets'])

        offsets_P = get_peaks(waves_dwt['ECG_P_Offsets'])
        offsets_R = get_peaks(waves_dwt['ECG_R_Offsets'])
        offsets_T = get_peaks(waves_dwt['ECG_T_Offsets'])

        signal = ecg_signal
        amp_R = get_amps(signal, peaks_R)
        amp_P = get_amps(signal, peaks_P)
        amp_Q = get_amps(signal, peaks_Q)
        amp_S = get_amps(signal, peaks_S)
        amp_T = get_amps(signal, peaks_T)

        length = len(peaks_R)

        peaks_P = fill_peaks(peaks_P, length)
        peaks_Q = fill_peaks(peaks_Q, length)
        peaks_R = fill_peaks(peaks_R, length)
        peaks_S = fill_peaks(peaks_S, length)
        peaks_T = fill_peaks(peaks_T, length)

        onsets_P = fill_peaks(onsets_P, length)
        onsets_R = fill_peaks(onsets_R, length)
        onsets_T = fill_peaks(onsets_T, length)

        offsets_P = fill_peaks(offsets_P, length)
        offsets_R = fill_peaks(offsets_R, length)
        offsets_T = fill_peaks(offsets_T, length)

        ######################### WAVEFORM ANALYSIS ##########################

        ############################# Intervals ##############################

        int_P = get_interval(onsets_P, offsets_P)
        int_T = get_interval(onsets_T, offsets_T)
        int_QRS = get_interval(onsets_R, offsets_R)

        # RR INTERVAL
        RR = []
        for j in range(1, len(peaks_R)):
            a = peaks_R
            RR = (a[j] - a[j - 1])
            int_RR.append(RR)

        ########################## Amplitudes ################################

        amp_P = fill_amps(amp_P, length)
        amp_R = fill_amps(amp_R, length)
        amp_Q = fill_amps(amp_Q, length)
        amp_S = fill_amps(amp_S, length)
        amp_T = fill_amps(amp_T, length)

        # Calculate absolute amplitude of each wave
        amp_P_absolute = max(amp_P) - min(amp_P)
        amp_Q_absolute = max(amp_Q) - min(amp_Q)
        amp_R_absolute = max(amp_R) - min(amp_R)
        amp_S_absolute = max(amp_S) - min(amp_S)
        amp_T_absolute = max(amp_T) - min(amp_T)

        # Calculate relative amplitude of R wave compared to P and T waves
        amp_R_relative_to_P = max(amp_R) / max(amp_P)
        amp_R_relative_to_T = max(amp_R) / max(amp_T)

        # Calculate statistical summaries for each wave
        mean_P, median_P, cv_P, std_dev_P, range_P = calculate_statistics(amp_P)
        mean_Q, median_Q, cv_Q, std_dev_Q, range_Q = calculate_statistics(amp_Q)
        mean_R, median_R, cv_R, std_dev_R, range_R = calculate_statistics(amp_R)
        mean_S, median_S, cv_S, std_dev_S, range_S = calculate_statistics(amp_S)
        mean_T, median_T, cv_T, std_dev_T, range_T = calculate_statistics(amp_T)

        ######################### HEARTRATE METRICS ##########################
        int_RR_sec = [interval / 1000 for interval in int_RR]
        avg_int_RR = sum(int_RR_sec) / len(int_RR)

        shortest_int_RR = min(int_RR_sec)
        longest_int_RR = max(int_RR_sec)
        avg_heartrate = 60 / avg_int_RR
        max_heartrate = 60 / shortest_int_RR
        min_heartrate = 60 / longest_int_RR

        sdnn = np.std(int_RR_sec)
        rmssd = np.sqrt(np.mean(np.square(np.diff(int_RR_sec))))
        pnn50 = np.sum(np.abs(np.diff(int_RR_sec)) > 0.05) / len(int_RR_sec) * 100

        #################### FREQUENCY DOMAIN FEATURES #######################

        sampling_rate = 500

        frequencies, psd = welch(ecg_signal, fs=sampling_rate, nperseg=1024, nfft=2048, scaling='density', window='hann', detrend='linear', average='mean')

        total_power = np.sum(psd)
        peak_frequency = frequencies[np.argmax(psd)]
        spectral_entropy = -np.sum(psd * np.log(psd))
        psd_threshold = max(psd) / 2
        above_threshold = psd > psd_threshold
        frequency_width = frequencies[above_threshold][-1] - frequencies[above_threshold][0]

        ######################## CREATING DICTIONARY ############################

        features_dict = {}
        features_dict["Patient_ID"] = patient
        features_dict["Age"] = np.float64(age[x])
        features_dict["Sex"] = sex[x]

        features_dict["HRM_Average_HR"] = avg_heartrate
        features_dict["HRM_Max_HR"] = max_heartrate
        features_dict["HRM_Min_HR"] = min_heartrate
        features_dict["HRM_Sdnn_RR"] = sdnn
        features_dict["HRM_RMSSD"] = rmssd
        features_dict["HRM_PNN50"] = pnn50

        features_dict["EWA_ABSAmp_P"] = amp_P_absolute
        features_dict["EWA_ABSAmp_Q"] = amp_Q_absolute
        features_dict["EWA_ABSAmp_R"] = amp_R_absolute
        features_dict["EWA_ABSAmp_S"] = amp_S_absolute
        features_dict["EWA_ABSAmp_T"] = amp_T_absolute

        features_dict["EWA_Amp_R_to_P"] = amp_R_relative_to_P
        features_dict["EWA_Amp_R_to_T"] = amp_R_relative_to_T

        features_dict["FA_Total_Power"] = total_power
        features_dict["FA_Peak_Frequency"] = peak_frequency
        features_dict["FA_Spectral_Entropy"] = spectral_entropy
        features_dict["FA_Frequency_Width"] = frequency_width
        
        features_dict["Mean_P"] = mean_P
        features_dict["Mean_Q"] = mean_Q
        features_dict["Mean_R"] = mean_R
        features_dict["Mean_S"] = mean_S
        features_dict["Mean_T"] = mean_T
        
        features_dict["Median_P"] = median_P
        features_dict["Median_Q"] = median_Q
        features_dict["Median_R"] = median_R
        features_dict["Median_S"] = median_S
        features_dict["Median_T"] = median_T
        
        features_dict["cv_P"] = cv_P # Coeffiient of variation
        features_dict["cv_Q"] = cv_P
        features_dict["cv_R"] = cv_P
        features_dict["cv_S"] = cv_P
        features_dict["cv_T"] = cv_P
        
        features_dict["Std_Dev_P"] = std_dev_P
        features_dict["Std_Dev_Q"] = std_dev_Q
        features_dict["Std_Dev_R"] = std_dev_R
        features_dict["Std_Dev_S"] = std_dev_S
        features_dict["Std_Dev_T"] = std_dev_T
        
        features_dict["Range_P"] = range_P
        features_dict["Range_Q"] = range_Q
        features_dict["Range_R"] = range_R
        features_dict["Range_S"] = range_S
        features_dict["Range_T"] = range_T

        features_dict["Diagnosis"] = diagnosis[x]
        
        # #Plotting the peaks
        # plot = nk.events_plot([waves_dwt['ECG_T_Peaks'], 
        #                         waves_dwt['ECG_P_Peaks'],
        #                         rpeaks['ECG_R_Peaks'],
        #                         waves_dwt['ECG_Q_Peaks'],
        #                         waves_dwt['ECG_S_Peaks']], ecg_signal)
        
        

        # #Plotting The P-Range, R-Range, and T-Range
        # plot = nk.events_plot([waves_dwt['ECG_T_Onsets'], 
        #                         waves_dwt['ECG_T_Offsets'], 
        #                         waves_dwt['ECG_P_Onsets'], 
        #                         waves_dwt['ECG_P_Offsets'],
        #                         waves_dwt['ECG_R_Onsets'], 
        #                         waves_dwt['ECG_R_Offsets']],ecg_signal)
        

        all_dict.append(features_dict)
        #all_list_dict.append(features_list_dict)
        all_ecg.append(ecg_signal)

    except (SkipFileWarning, ValueError, IndexError, KeyError) as e:
        print(f"Skipping file for patient {names[x]} due to error: {e}")
        # Move the file to the error directory
        filename = f"{names[x]}.hea"
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        shutil.move(source_file, destination_file)
        continue
