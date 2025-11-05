import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

from preprocess import *
import FeatureExtractor as FE
import WindowIterator as WI

feature_dict = {'EDA': {'SCL' : ['mean', 'std', 'slope'], 'SCR' : ['mean', 'max', 'counts']},
				'ECG' : ['mean_rate', 'std_rate', 'inter_peak_time', 'RMSSD', 'SDNN', 'HF', 'VHF', 'TP', 'HF_norm'],
				'Temp': ['mean', 'std', 'min','max', 'slope'],
				 'Resp': {'Amp' : ['mean_rate', 'mean', 'std', 'SDNN'], 'RVT' : ['mean']}}
sampling_freq_dict = {'EDA': 700, 'ECG': 700, 'Temp': 700, 'Resp':700}
modality = 'chest'

# feature_dict = {'BVP' : ['mean_rate', 'std_rate', 'inter_peak_time', 'RMSSD', 'SDNN'],'Temp': ['mean', 'std', 'min','max', 'slope']}
# sampling_freq_dict = {'EDA' : 4, 'BVP':64, 'Temp':4}
# modality = 'wrist'


feature_extractor = FE.FeatureExtractor(feature_dict, sampling_freq_dict)

window_in_sec = 60
shift_in_sec = 5	

train_df, test_df = get_train_test_data(test_subjects=[14,15,16,17], modality=modality, window_in_sec=window_in_sec, shift_in_sec=shift_in_sec,
										 feature_dict=feature_dict, sampling_freq_dict=sampling_freq_dict, calibration_frac=0.25)

filename = f'_{modality}_{window_in_sec}_{shift_in_sec}.csv'

train_df.to_csv('train' + filename, index=False)
test_df.to_csv('test' + filename, index=False)



# train_df = get_subject_features(subject_idx=2, modality='chest', window_in_sec=30, shift_in_sec=5,
# 								 feature_extractor=feature_extractor, sampling_freq_dict=sampling_freq_dict, calibration_frac=1, include_calibration=True)

# train_df_1 = train_df.dropna(axis=1, how='all')

# cols = [c for c in train_df.columns if c not in train_df_1.columns ]


# print(train_df.head())
# print(train_df_1.head())
# print(cols)