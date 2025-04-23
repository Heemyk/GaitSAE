import pywt
import numpy as np

def preprocess_gait_signals(data_dict):
    """
    Preprocess the gait signals using DWT
    """
    # Using PRO data since it's already normalized and filtered
    processed_data = {}
    wavelet = 'db5'  # Daubechies 5 wavelet
    level = 4  # Decomposition level
    
    for key, data in data_dict.items():
        if 'PRO' in key:  # Only use processed data
            # Each row is a stance phase with 101 points
            coeffs_list = []
            for i, stance_phase in enumerate(data.values):  # Assuming time series are in rows
                # Perform DWT
                coeffs = pywt.wavedec(stance_phase, wavelet, level=level)
                # Concatenate coefficients
                coeffs_flat = np.concatenate([c for c in coeffs])
                coeffs_list.append(coeffs_flat)

            processed_data[key] = np.array(coeffs_list)
    
    return processed_data
