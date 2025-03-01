import numpy as np
import scipy
import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import savgol_filter
import csv
import neurokit2 as nk
from scipy.signal import welch


def get_R_amps(signal, peaks_X):
    peaks = peaks_X
    amp = []
    for location in peaks:
        if np.isnan(location):
            amp.append(np.nan)
        else:
            location = int(location / 2)
            if signal[location] > 100:
                amp.append(signal[location])
    return amp

def get_amps(signal, peaks_X):
    amp=[]
    for location in peaks_X:
        if np.isnan(location):
            amp.append(np.nan)
        else:
            location=int(location/2)
            amp.append(signal[location])
    return amp

def get_peaks(parameter):
    X=[]
    X=[i*2 for i in parameter]
    #X = [y for y in X if not math.isnan(y)]
    return X

def get_interval(list1, list2):
    seg=[]
    a= list1
    b=list2
    seg=[x - y for x, y in zip(b, a)]
    return seg

def custom_function(original_list):
    # Perform some operation on the value
    converted_list = [np.array(item) if isinstance(item, list) else item for item in original_list]
    converted_list = [np.array(item) if isinstance(item, int) else item for item in original_list]
    return converted_list

def fill_peaks(lst, desired_length):
    # Calculate the average difference between consecutive non-NaN values
    non_nan_values = [value for value in lst if not np.isnan(value)]
    
    if len(non_nan_values) <= 1:
        raise ValueError("At least two non-NaN values are required for interpolation.")
    
    avg_difference = (non_nan_values[-1] - non_nan_values[0]) / (len(non_nan_values) - 1)

    # Replace NaN values based on specified rules
    for i in range(len(lst)):
        if np.isnan(lst[i]):
            if i == 0:
                # Handle the case where the first value is NaN
                lst[i] = round(max(0, non_nan_values[0] - avg_difference))
            else:
                # Interpolate based on the previous non-NaN value
                lst[i] = round(max(0, lst[i - 1] + avg_difference))
            # Ensure that the maximum value is 9998
            lst[i] = round(min(9998, lst[i]))

    # Adjust the length of the list if necessary
    if len(lst) < desired_length:
        lst.extend([min(9998, lst[-1])] * (desired_length - len(lst)))
    elif len(lst) > desired_length:
        lst = lst[:desired_length]

    return lst

def fill_amps(input_list, desired_length):
    # Remove NaN values and calculate the average of non-NaN values
    non_nan_values = [value for value in input_list if not np.isnan(value)]
    average = np.mean(non_nan_values)
    average= round(average)

    # Replace NaN values with the calculated average
    result_list = [average if np.isnan(value) else value for value in input_list]

    # Ensure the list has the desired length
    while len(result_list) < desired_length:
        result_list.append(average)

    return result_list[:desired_length]


def detect_abnormalities(ecg_signal):
    abnormal_waveforms = []
    st_elevation_threshold = 0.1  # Example threshold for ST-segment elevation
    st_depression_threshold = -0.1  # Example threshold for ST-segment depression
    # Iterate over the ECG signal
    for i in range(len(ecg_signal)):
        # Check for ST-segment elevation
        if ecg_signal[i] > st_elevation_threshold:
            abnormal_waveforms.append(("ST-elevation", i))  # Add abnormality to the list
        # Check for ST-segment depression
        elif ecg_signal[i] < st_depression_threshold:
            abnormal_waveforms.append(("ST-depression", i))  # Add abnormality to the list
    return abnormal_waveforms


def calculate_statistics(amp):
    mean_amp = np.mean(amp)
    median_amp = np.median(amp)
    std_dev_amp = np.std(amp)
    range_amp = max(amp) - min(amp)
    cv = (std_dev_amp / mean_amp) * 100 if mean_amp != 0 else 0  # Avoid division by zero
    return mean_amp, median_amp, cv, std_dev_amp, range_amp


def get_diagnosis_names(diagnosis_codes):
    
    if isinstance(diagnosis_codes, list):
        return [diagnosis_mapping.get(int(code), 'Unknown') for code in diagnosis_codes]
    elif isinstance(diagnosis_codes, str):
        codes_list = diagnosis_codes.split(',')
        return [diagnosis_mapping.get(int(code.strip()), 'Unknown') for code in codes_list]
    else:
        return 'Unknown'


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

def pre_processing(ecg_array, notch_frequency, notch_bandwidth, baseline_window_size, fs):
    
    # Designing a notch filter to remove powerline interference (50-60Hz)
    nyquist_frequency = 0.5 * fs
    notch_freq = notch_frequency / nyquist_frequency
    notch_bw = notch_bandwidth / nyquist_frequency
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, notch_bw)

    # Applying the notch filter to each lead of the ECG array
    ecg_array_filtered = scipy.signal.lfilter(b_notch, a_notch, ecg_array)
    
    # Baseline wander correction using moving average
    moving_avg = np.convolve(ecg_array_filtered, np.ones(baseline_window_size)/baseline_window_size, mode='same')
    corrected_ecg_array = ecg_array_filtered - moving_avg
    
    # Apply smoothing filter to each lead of the corrected ECG array
    window_size = 25  # Should be an odd number
    order = 2  # Polynomial order
    smoothed_ecg_array = savgol_filter(corrected_ecg_array, window_size, order)
    
    # Calculate percentile values for clipping
    clip_min = np.percentile(smoothed_ecg_array, 1)
    clip_max = np.percentile(smoothed_ecg_array, 99)

    # Clip values to the calculated percentiles
    clipped_ecg_array = np.clip(smoothed_ecg_array, clip_min, clip_max)

    # Normalize the clipped array
    min_value = np.min(clipped_ecg_array)
    max_value = np.max(clipped_ecg_array)
    ecg_normalized_array = (clipped_ecg_array - min_value) / (max_value - min_value)
    
    return ecg_normalized_array

    

def get_features(signal,age):
    fs=500
    num_samples = signal.shape
    
    time = len(signal)/fs
    peaks_R,peaks_P,peaks_Q,peaks_S,peaks_T,onsets_P,onsets_R,onsets_T,offsets_P,offsets_R,offsets_T,seg_PR,seg_ST,int_PR,int_ST,int_QT,int_RR,int_QRS,amp_R,amp_P,amp_Q,amp_S,amp_T=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    
    # Extracting R-peaks locations
    _, rpeaks= nk.ecg_peaks(signal, sampling_rate=fs)
    R= rpeaks['ECG_R_Peaks'].tolist()
    R= [j*2 for j in R]
    #peaks_R[lead]=R
    
    # Delineating the ECG signal for peaks
    _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method="peak")
    
    # Getting the wave boundaries
    signal_cwt, waves_cwt = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method="cwt")
    signal_dwt, waves_dwt = nk.ecg_delineate(signal, rpeaks, sampling_rate=fs, method="dwt")
    
    peaks_P=get_peaks(waves_dwt['ECG_P_Peaks'])
    peaks_Q=get_peaks(waves_dwt['ECG_Q_Peaks'])
    peaks_R=get_peaks(rpeaks['ECG_R_Peaks'])
    peaks_S=get_peaks(waves_dwt['ECG_S_Peaks'])
    peaks_T=get_peaks(waves_dwt['ECG_T_Peaks'])
    
    onsets_P=get_peaks(waves_dwt['ECG_P_Onsets'])
    onsets_R=get_peaks(waves_dwt['ECG_R_Onsets'])
    onsets_T=get_peaks(waves_dwt['ECG_T_Onsets'])
    
    offsets_P=get_peaks(waves_dwt['ECG_P_Offsets'])
    offsets_R=get_peaks(waves_dwt['ECG_R_Offsets'])
    offsets_T=get_peaks(waves_dwt['ECG_T_Offsets'])
    
    signal= signal
    amp_R=get_amps(signal, peaks_R)
    amp_P=get_amps(signal, peaks_P)
    amp_Q=get_amps(signal, peaks_Q)
    amp_S=get_amps(signal, peaks_S)
    amp_T=get_amps(signal, peaks_T)
    
    length= len(peaks_R) 
    
    peaks_P=fill_peaks(peaks_P,length)
    peaks_Q=fill_peaks(peaks_Q,length)
    peaks_R=fill_peaks(peaks_R,length)
    peaks_S=fill_peaks(peaks_S,length)
    peaks_T=fill_peaks(peaks_T,length)
    
    onsets_P=fill_peaks(onsets_P,length)
    onsets_R=fill_peaks(onsets_R,length)
    onsets_T=fill_peaks(onsets_T,length)
    
    offsets_P=fill_peaks(offsets_P,length)
    offsets_R=fill_peaks(offsets_R,length)
    offsets_T=fill_peaks(offsets_T,length)
    #FI
    
    
    ######################### WAVEFORM ANALYSIS ##########################
    
    ############################# Intervals ##############################
    
    int_P= get_interval(onsets_P,offsets_P)
    int_T= get_interval(onsets_T,offsets_T)
    int_QRS= get_interval(onsets_R,offsets_R)
    
     
    #RR INTERVAL 
    RR=[]
    for j in range(1, len(peaks_R)):
        a=peaks_R
        #int_RR.append([x - y for x, y in zip(a[j], a[j-1])])
        RR=(a[j]-a[j-1]) 
        int_RR.append(RR)
        
    ########################## Amplitudes ################################
    
    amp_P=fill_amps(amp_P, length) #atrial depolarisation
    amp_R=fill_amps(amp_R, length) #Initial ventricular depolarisation
    amp_Q=fill_amps(amp_Q, length) #peak of ventricular depolarisation
    amp_S=fill_amps(amp_S, length) #Downward deflection following R wave
    amp_T=fill_amps(amp_T, length) #Ventricular polarisation
    
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
    
    
    # statistics for each wave (mean, std_dev, median, range, coefficient of variability [cv])
    mean_P, median_P, cv_P, std_dev_P, range_P = calculate_statistics(amp_P)
    mean_Q, median_Q, cv_Q, std_dev_Q, range_Q = calculate_statistics(amp_Q)
    mean_R, median_R, cv_R, std_dev_R, range_R = calculate_statistics(amp_R)
    mean_S, median_S, cv_S, std_dev_S, range_S = calculate_statistics(amp_S)
    mean_T, median_T, cv_T, std_dev_T, range_T = calculate_statistics(amp_T)
    
    
    # do detection of abnormal waveforms later
    
    ## Print out the results
    # print("Statistics for P wave:")
    # print("Mean:", mean_P)
    # print("Median:", median_P)
    # print("Standard Deviation:", std_dev_P)
    # print("Range:", range_P)
    # print()

        
    ######################################################################
    
    ######################### HEARTRATE METRICS ##########################
    int_RR_sec= [interval/1000 for interval in int_RR] #converting the intervals into seconds
    avg_int_RR = (sum(int_RR_sec)) / len(int_RR) #interval in seconds
    
    shortest_int_RR = min(int_RR_sec)
    longest_int_RR = max(int_RR_sec)
    avg_heartrate = 60 / avg_int_RR  # Beats per minute
    max_heartrate = 60 / shortest_int_RR  # Beats per minute
    min_heartrate = 60 / longest_int_RR  # Beats per minute
    
    sdnn = np.std(int_RR_sec)  #Standard deviation of RR intervals
    rmssd = np.sqrt(np.mean(np.square(np.diff(int_RR_sec)))) #RMS of RR Intervals
    pnn50 = np.sum(np.abs(np.diff(int_RR_sec)) > 0.05) / len(int_RR_sec) * 100 # Percentage of RR intervals differing by more than 50 ms (pNN50)
    
    ######################################################################
    #################### FREQUENCY DOMAIN FEATURES #######################

    sampling_rate = 500  # Updated sampling rate (Hz)
    
    # Calculate power spectral density (PSD) using Welch's method
    frequencies, psd = welch(signal, fs=sampling_rate, nperseg=1024, nfft=2048, scaling='density', window='hann', detrend='linear', average='mean')
    
    # Compute total power
    total_power = np.sum(psd)
    
    # Compute peak frequency
    peak_frequency = frequencies[np.argmax(psd)]
    
    # Compute spectral entropy
    spectral_entropy = -np.sum(psd * np.log(psd))
    
    # Compute frequency variability (width of the PSD)
    psd_threshold = max(psd) / 2  # Threshold for half maximum power
    above_threshold = psd > psd_threshold
    frequency_width = frequencies[above_threshold][-1] - frequencies[above_threshold][0]

    # #Plot the power spectral density (PSD)
    # plt.figure(figsize=(10, 6))
    # plt.plot(frequencies, psd)
    # plt.title('Power Spectral Density (PSD) of ECG Signal')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power/Frequency (dB/Hz)')
    # plt.grid(True)
    # plt.xlim(0, 40)  # Adjust the limits as needed
    # plt.show()

   
   
   ######################## CREATING DICTIONARY ############################
   
    # Initialize an empty dictionary to store the features
    features_dict = {}
    # features_dict["Patient_ID"] = patient
    features_dict["Age"] = np.float64(age)
    # features_dict["Sex"] = sex[x]

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

    #features_dict["Diagnosis"] = diagnosis[x]
    #features_dict["Heart_Rate"] = Heartrate

    return features_dict



