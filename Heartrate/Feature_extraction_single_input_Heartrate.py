# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Heartrate_Initial
import Functions_HR
import math
import os
import statsmodels.api as sm
from scipy.signal import spectrogram
from Functions_HR import get_R_amps, get_amps, get_peaks, get_interval, custom_function, fill_peaks, fill_amps


all_dict=[]
all_ecg=[]
all_gender=[]

# Retrieve ECG data
list_of_patient_data= Heartrate_Initial.ecg_final
names= Heartrate_Initial.root
diagnosis= Heartrate_Initial.patients_with_single_diagnosis

flat_diagnoses = [diag for sublist in diagnosis for diag in sublist]
unique_diagnoses = set(flat_diagnoses)
unique_diagnoses_count = len(unique_diagnoses)

sex= Heartrate_Initial.sex
age= Heartrate_Initial.age

for x, patient_data in enumerate(list_of_patient_data):
    ecg_signal= patient_data[0]
    #ecg_orig_signal= Heartrate_Initial.ecg_array[0]
    #ecg_signal=nk.ecg_clean(ecg_signal, sampling_rate=500, method='neurokit')
    fs=500
    patient=names[x]
    num_samples = ecg_signal.shape
    
    time = len(ecg_signal)/fs
    peaks_R,peaks_P,peaks_Q,peaks_S,peaks_T,onsets_P,onsets_R,onsets_T,offsets_P,offsets_R,offsets_T,seg_PR,seg_ST,int_PR,int_ST,int_QT,int_RR,int_QRS,amp_R,amp_P,amp_Q,amp_S,amp_T=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    
        
    # Extracting R-peaks locations
    _, rpeaks= nk.ecg_peaks(ecg_signal, sampling_rate=fs)
    R= rpeaks['ECG_R_Peaks'].tolist()
    R= [j*2 for j in R]
    #peaks_R[lead]=R
    
    
    # Delineating the ECG signal for peaks
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak")
    
    # Getting the wave boundaries
    signal_cwt, waves_cwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="cwt")
    signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="dwt")
    
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
    
    signal= ecg_signal
    amp_R=get_amps(signal, peaks_R)
    amp_P=get_amps(signal, peaks_P)
    amp_Q=get_amps(signal, peaks_Q)
    amp_S=get_amps(signal, peaks_S)
    amp_T=get_amps(signal, peaks_T)
    
    # #Plotting the peaks
    # plot = nk.events_plot([waves_dwt['ECG_T_Peaks'], 
    #                         waves_dwt['ECG_P_Peaks'],
    #                         rpeaks['ECG_R_Peaks'],
    #                         waves_dwt['ECG_Q_Peaks'],
    #                         waves_dwt['ECG_S_Peaks']], ecg_signal)
    
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
    
    amp_R=fill_amps(amp_R, length)
    amp_P=fill_amps(amp_P, length)
    amp_Q=fill_amps(amp_Q, length)
    amp_S=fill_amps(amp_S, length)
    amp_T=fill_amps(amp_T, length)
    
    
    #Finding the average of each feature
    seg_PR= get_interval(offsets_P,onsets_R)
    seg_ST= get_interval(offsets_R,onsets_T)
    int_PR= get_interval(onsets_P,onsets_R)
    int_ST= get_interval(offsets_R,offsets_T)
    int_QT= get_interval(onsets_R,offsets_T)
    int_QRS= get_interval(onsets_R,offsets_R)
    
     
    #RR INTERVAL 
    RR=[]
    for j in range(1, len(peaks_R)):
        a=peaks_R
        #int_RR.append([x - y for x, y in zip(a[j], a[j-1])])
        RR=(a[j]-a[j-1]) 
        int_RR.append(RR)
    
    # while len(int_RR) < length:
    #     int_RR.append(np.nan)
    #int_RR= fill_amps(int_RR, length)
    #HEARTRATE
    Heart_rate=len(peaks_R)*6
    
    
    Filtered_signal = ecg_signal.reshape(-1, 1)
    
    av_seg_PR= round(sum(seg_PR) / len(seg_PR), 1)
    av_seg_ST= round(sum(seg_ST) / len(seg_ST), 1)
    av_int_PR= round(sum(int_PR) / len(int_PR), 1)
    av_int_ST= round(sum(int_ST) / len(int_ST), 1)
    av_int_QT= round(sum(int_QT) / len(int_QT), 1)
    av_int_RR= round(sum(int_RR) / len(int_RR), 1)
    av_int_QRS= round(sum(int_QRS) / len(int_QRS), 1)
    av_amp_P= round(sum(amp_P) / len(amp_P), 1)
    av_amp_Q= round(sum(amp_Q) / len(amp_Q), 1)
    av_amp_R= round(sum(amp_R) / len(amp_R), 1)
    av_amp_S= round(sum(amp_S) / len(amp_S), 1)
    av_amp_T= round(sum(amp_T) / len(amp_T), 1)

        
    #Plotting The P-Range, R-Range, and T-Range
    plot = nk.events_plot([waves_dwt['ECG_T_Onsets'], 
                            waves_dwt['ECG_T_Offsets'], 
                            waves_dwt['ECG_P_Onsets'], 
                            waves_dwt['ECG_P_Offsets'],
                            waves_dwt['ECG_R_Onsets'], 
                            waves_dwt['ECG_R_Offsets']],ecg_signal)
        
    # Initialize an empty dictionary to store the features
    features_dict = {}
    #diag= str(diagnosis[x])
    # Add the features lists to the dictionary
    features_dict["PR_Segments"] = av_seg_PR
    features_dict["ST_Segments"] = av_seg_ST
    features_dict["PR_Intervals"] = av_int_PR
    features_dict["ST_Intervals"] = av_int_ST
    features_dict["QT_Intervals"] = av_int_QT
    features_dict["RR_Intervals"] = av_int_RR
    features_dict["QRS_Intervals"] = av_int_QRS
    features_dict["Amplitudes_P"] = av_amp_P
    features_dict["Amplitudes_Q"] = av_amp_Q
    features_dict["Amplitudes_R"] = av_amp_R
    features_dict["Amplitudes_S"] = av_amp_S
    features_dict["Amplitudes_T"] = av_amp_T
    features_dict["Heart_Rate"] = Heart_rate
    features_dict["Age"]= int(age[x])
    features_dict["Sex"]= sex[x]
    features_dict["Diagnosis"]= diagnosis[x]
    
    
    all_dict.append(features_dict)
    all_ecg.append(ecg_signal)
    
    
    
    # df = pd.DataFrame(features_dict)
    
    # folder_path =  '/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/DATA'
    # csv_file_path = os.path.join(folder_path, f"{names[x]}.csv")
    # npy_file_path= os.path.join(folder_path, f"{names[x]}.npy")
    # # Save the DataFrame to a CSV file with the constructed name
    
    # df.to_csv(csv_file_path, index=False)
    # np.save(npy_file_path, Filtered_signal)
    
    
    # # loaded= np.load('/Users/aashika/Desktop/Classes/Semester-3/Thesis/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/DATA/JS00095.npy')
