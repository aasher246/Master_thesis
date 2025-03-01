import numpy as np

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