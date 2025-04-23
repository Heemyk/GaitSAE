import pandas as pd
import os
import numpy as np

# For Colab
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_ROOT = '/content/drive/MyDrive/GaitSAE'  # Change this to your Drive folder path

# For local
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def add_dataset_id(data):
    # for the GaitRec data we will define the dataset ID as 0
    data['DATASET_ID'] = 0
    return data

def make_unique_id(data1, data2):
    # SUBJECT_ID and SESSION_ID of data2 are changed to ensure their uniqueness (by adding the
    # maximum ID to the IDs of the GaitRec dataset)
    
    max_id = np.max(data1['SUBJECT_ID'].values)
    data2['SUBJECT_ID'] = data2['SUBJECT_ID']+max_id
    
    max_id = np.max(data1['SESSION_ID'].values)
    data2['SESSION_ID'] = data2['SESSION_ID']+max_id
    
    return data2 

def merge_data(data1, data2):
    # prior to merging we need to add a DATASET_ID for data2 and change the
    # SUBJECT_ID and SESSION_ID to ensure their uniqueness
    
    data2 = add_dataset_id(data2)
    data2 = make_unique_id(data1,data2)
    data = pd.concat([data1, data2], ignore_index=True, sort=False)
    return data

def load_gait_data(use_gaitrec=False):
    """
    Load gait data from GutenbergGaitDatabase and optionally GaitRec.
    
    Args:
        use_gaitrec (bool): Whether to include GaitRec data. Defaults to False.
    
    Returns:
        dict: Dictionary containing all loaded dataframes
    """
    # Initialize dictionary to store all dataframes
    data_dict = {}
    
    # Path to GutenbergGaitDatabase
    path = os.path.join(DATA_ROOT, 'data', 'GutenbergGaitDatabase')
    
    # Left lower extremity
    data_dict['GRF_F_V_PRO_left'] = pd.read_csv(os.path.join(path,'GRF_F_V_PRO_left.csv'))
    data_dict['GRF_F_V_RAW_left'] = pd.read_csv(os.path.join(path,'GRF_F_V_RAW_left.csv'))
    
    data_dict['GRF_F_AP_PRO_left'] = pd.read_csv(os.path.join(path,'GRF_F_AP_PRO_left.csv'))
    data_dict['GRF_F_AP_RAW_left'] = pd.read_csv(os.path.join(path,'GRF_F_AP_RAW_left.csv'))
    
    data_dict['GRF_F_ML_PRO_left'] = pd.read_csv(os.path.join(path,'GRF_F_ML_PRO_left.csv'))
    data_dict['GRF_F_ML_RAW_left'] = pd.read_csv(os.path.join(path,'GRF_F_ML_RAW_left.csv'))
    
    data_dict['GRF_COP_AP_PRO_left'] = pd.read_csv(os.path.join(path,'GRF_COP_AP_PRO_left.csv'))
    data_dict['GRF_COP_AP_RAW_left'] = pd.read_csv(os.path.join(path,'GRF_COP_AP_RAW_left.csv'))
    
    data_dict['GRF_COP_ML_PRO_left'] = pd.read_csv(os.path.join(path,'GRF_COP_ML_PRO_left.csv'))
    data_dict['GRF_COP_ML_RAW_left'] = pd.read_csv(os.path.join(path,'GRF_COP_ML_RAW_left.csv'))
    
    # Right lower extremity
    data_dict['GRF_F_V_PRO_right'] = pd.read_csv(os.path.join(path,'GRF_F_V_PRO_right.csv'))
    data_dict['GRF_F_V_RAW_right'] = pd.read_csv(os.path.join(path,'GRF_F_V_RAW_right.csv'))
    
    data_dict['GRF_F_AP_PRO_right'] = pd.read_csv(os.path.join(path,'GRF_F_AP_PRO_right.csv'))
    data_dict['GRF_F_AP_RAW_right'] = pd.read_csv(os.path.join(path,'GRF_F_AP_RAW_right.csv'))
    
    data_dict['GRF_F_ML_PRO_right'] = pd.read_csv(os.path.join(path,'GRF_F_ML_PRO_right.csv'))
    data_dict['GRF_F_ML_RAW_right'] = pd.read_csv(os.path.join(path,'GRF_F_ML_RAW_right.csv'))
    
    data_dict['GRF_COP_AP_PRO_right'] = pd.read_csv(os.path.join(path,'GRF_COP_AP_PRO_right.csv'))
    data_dict['GRF_COP_AP_RAW_right'] = pd.read_csv(os.path.join(path,'GRF_COP_AP_RAW_right.csv'))
    
    data_dict['GRF_COP_ML_PRO_right'] = pd.read_csv(os.path.join(path,'GRF_COP_ML_PRO_right.csv'))
    data_dict['GRF_COP_ML_RAW_right'] = pd.read_csv(os.path.join(path,'GRF_COP_ML_RAW_right.csv'))
    
    # Walking Speed
    data_dict['GRF_walking_speed'] = pd.read_csv(os.path.join(path,'GRF_walking_speed.csv'))
    
    # Metadata
    data_dict['GRF_metadata'] = pd.read_csv(os.path.join(path,'GRF_metadata.csv'))
    
    if use_gaitrec:
        # Path to GaitRec
        path = os.path.join(DATA_ROOT, 'data', 'GaitRec')
        
        # Left lower extremity
        data_dict['GRF_F_V_PRO_left'] = merge_data(data_dict['GRF_F_V_PRO_left'], 
                                                  pd.read_csv(os.path.join(path,'GRF_F_V_PRO_left.csv')))
        data_dict['GRF_F_V_RAW_left'] = merge_data(data_dict['GRF_F_V_RAW_left'], 
                                                  pd.read_csv(os.path.join(path,'GRF_F_V_RAW_left.csv')))
        
        data_dict['GRF_F_AP_PRO_left'] = merge_data(data_dict['GRF_F_AP_PRO_left'], 
                                                   pd.read_csv(os.path.join(path,'GRF_F_AP_PRO_left.csv')))
        data_dict['GRF_F_AP_RAW_left'] = merge_data(data_dict['GRF_F_AP_RAW_left'], 
                                                   pd.read_csv(os.path.join(path,'GRF_F_AP_RAW_left.csv')))
        
        data_dict['GRF_F_ML_PRO_left'] = merge_data(data_dict['GRF_F_ML_PRO_left'], 
                                                   pd.read_csv(os.path.join(path,'GRF_F_ML_PRO_left.csv')))
        data_dict['GRF_F_ML_RAW_left'] = merge_data(data_dict['GRF_F_ML_RAW_left'], 
                                                   pd.read_csv(os.path.join(path,'GRF_F_ML_RAW_left.csv')))
        
        data_dict['GRF_COP_AP_PRO_left'] = merge_data(data_dict['GRF_COP_AP_PRO_left'], 
                                                     pd.read_csv(os.path.join(path,'GRF_COP_AP_PRO_left.csv')))
        data_dict['GRF_COP_AP_RAW_left'] = merge_data(data_dict['GRF_COP_AP_RAW_left'], 
                                                     pd.read_csv(os.path.join(path,'GRF_COP_AP_RAW_left.csv')))
        
        data_dict['GRF_COP_ML_PRO_left'] = merge_data(data_dict['GRF_COP_ML_PRO_left'], 
                                                     pd.read_csv(os.path.join(path,'GRF_COP_ML_PRO_left.csv')))
        data_dict['GRF_COP_ML_RAW_left'] = merge_data(data_dict['GRF_COP_ML_RAW_left'], 
                                                     pd.read_csv(os.path.join(path,'GRF_COP_ML_RAW_left.csv')))
        
        # Right lower extremity
        data_dict['GRF_F_V_PRO_right'] = merge_data(data_dict['GRF_F_V_PRO_right'], 
                                                   pd.read_csv(os.path.join(path,'GRF_F_V_PRO_right.csv')))
        data_dict['GRF_F_V_RAW_right'] = merge_data(data_dict['GRF_F_V_RAW_right'], 
                                                   pd.read_csv(os.path.join(path,'GRF_F_V_RAW_right.csv')))
        
        data_dict['GRF_F_AP_PRO_right'] = merge_data(data_dict['GRF_F_AP_PRO_right'], 
                                                    pd.read_csv(os.path.join(path,'GRF_F_AP_PRO_right.csv')))
        data_dict['GRF_F_AP_RAW_right'] = merge_data(data_dict['GRF_F_AP_RAW_right'], 
                                                    pd.read_csv(os.path.join(path,'GRF_F_AP_RAW_right.csv')))
        
        data_dict['GRF_F_ML_PRO_right'] = merge_data(data_dict['GRF_F_ML_PRO_right'], 
                                                    pd.read_csv(os.path.join(path,'GRF_F_ML_PRO_right.csv')))
        data_dict['GRF_F_ML_RAW_right'] = merge_data(data_dict['GRF_F_ML_RAW_right'], 
                                                    pd.read_csv(os.path.join(path,'GRF_F_ML_RAW_right.csv')))
        
        data_dict['GRF_COP_AP_PRO_right'] = merge_data(data_dict['GRF_COP_AP_PRO_right'], 
                                                      pd.read_csv(os.path.join(path,'GRF_COP_AP_PRO_right.csv')))
        data_dict['GRF_COP_AP_RAW_right'] = merge_data(data_dict['GRF_COP_AP_RAW_right'], 
                                                      pd.read_csv(os.path.join(path,'GRF_COP_AP_RAW_right.csv')))
        
        data_dict['GRF_COP_ML_PRO_right'] = merge_data(data_dict['GRF_COP_ML_PRO_right'], 
                                                      pd.read_csv(os.path.join(path,'GRF_COP_ML_PRO_right.csv')))
        data_dict['GRF_COP_ML_RAW_right'] = merge_data(data_dict['GRF_COP_ML_RAW_right'], 
                                                      pd.read_csv(os.path.join(path,'GRF_COP_ML_RAW_right.csv')))
        
        # Walking Speed: is not specified for GaitRec dataset (we will add NaNs for the GaitRec data)
        data_dict['GRF_walking_speed'] = merge_data(data_dict['GRF_walking_speed'], 
                                                  pd.read_csv(os.path.join(path,'GRF_F_V_PRO_left.csv')).iloc[:,0:3])
        
        # Metadata
        data_dict['GRF_metadata'] = merge_data(data_dict['GRF_metadata'], 
                                             pd.read_csv(os.path.join(path,'GRF_metadata.csv')))
    
    return data_dict

# Example usage when running this file directly
if __name__ == "__main__":
    data = load_gait_data(use_gaitrec=False)
    print("Loaded data keys:", list(data.keys()))

