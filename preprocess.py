import numpy as np
import neurokit2 as nk
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import FeatureExtractor as FE
import WindowIterator as WI


	


def get_subject_features(subject_idx, modality, window_in_sec, shift_in_sec, feature_extractor, sampling_freq_dict, calibration_frac=1, include_calibration=True):
	'''
	This function obtains all the features for a given subject (denoted by their index) using a rolling window approach.

	Args:

	subject_idx (int): The index of the subject. Has to be in the set {2,3,4,5,6,7,8,9,10,11,13,14,15,16,17}.
	window_in_sec (float):  The length of the window (in seconds) to be used to create a single feature vector
	shift_in_sec (float): The time by which the window must be rolled to obtain the next data point.
	sampling_freq  (float): The sampling frequency of the signal in Hz. Assumed to be common across signals.
	calibration_frac (float): The fraction of the baseline data to be used for computing the mean and variance required for calibration
	include_calibration (bool): Denotes where to include the data used for calibration in the generated features

	Returns:

	dataset (pd.DataFrame): The dataset corresponding to the subject consisting of baseline data (label = 0) followed by 
							the stress data (label = 1). This contains all the windows arranged chronologically.


	'''
	
	# window_size = int(round(window_in_sec * sampling_freq))
	# shift = int(round(shift_in_sec * sampling_freq))
	
	# with open(f'./WESAD/S{subject_idx}/S{subject_idx}.pkl', 'rb') as f:
	# 	data = pickle.load(f, encoding="latin1")
	
	# signal = data['signal']['chest']
	# baseline_labels = data['label'] == 1
	# stress_labels = data['label'] == 2
	
	# baseline_signal = {}
	# stress_signal = {}
	
	# for k in ['EDA', 'ECG', 'Temp', 'Resp']:
		
	# 	baseline_signal[k] = signal[k][baseline_labels].squeeze()
	# 	stress_signal[k] = signal[k][stress_labels].squeeze()

	# baseline_df = pd.DataFrame(baseline_signal)
	# stress_df = pd.DataFrame(stress_signal)

	baseline_df, stress_df = get_separated_data(subject_idx, modality, sampling_freq_dict)

	baseline_iterator = WI.DataIterator(baseline_df, window_in_sec, shift_in_sec, sampling_freq_dict)
	stress_iterator = WI.DataIterator(stress_df, window_in_sec, shift_in_sec, sampling_freq_dict)

	baseline_feature_mat = get_feature_mat(baseline_df, baseline_iterator, feature_extractor)
	stress_feature_mat = get_feature_mat(stress_df, stress_iterator, feature_extractor)

	max_calibration_idx = int(round(calibration_frac*baseline_feature_mat.shape[0]))

	baseline_mean = baseline_feature_mat[:max_calibration_idx, :].mean(axis=0)
	baseline_std = baseline_feature_mat[:max_calibration_idx, :].std(axis=0)

	baseline_feature_mat_normalized = baseline_feature_mat - np.tile(baseline_mean, [baseline_feature_mat.shape[0], 1])
	baseline_feature_mat_normalized = baseline_feature_mat_normalized/np.tile(baseline_std, [baseline_feature_mat.shape[0], 1])

	stress_feature_mat_normalized = stress_feature_mat - np.tile(baseline_mean, [stress_feature_mat.shape[0], 1])
	stresss_feature_mat_normalized = stress_feature_mat_normalized/np.tile(baseline_std, [stress_feature_mat.shape[0], 1])

	if include_calibration:
		baseline_normalized_df = pd.DataFrame(baseline_feature_mat_normalized, columns=feature_extractor.feature_name_list)
	else:
		baseline_normalized_df = pd.DataFrame(baseline_feature_mat_normalized[max_calibration_idx:, :], columns=feature_extractor.feature_name_list)
	stress_normalized_df = pd.DataFrame(stress_feature_mat_normalized, columns=feature_extractor.feature_name_list)


	baseline_normalized_df['label'] = 0
	stress_normalized_df['label'] = 1

	return pd.concat([baseline_normalized_df, stress_normalized_df], ignore_index=True)


def get_separated_data(subject_idx, modality, sampling_freq_dict, label_samp_freq=700):

	with open(f'./WESAD/S{subject_idx}/S{subject_idx}.pkl', 'rb') as f:
		data = pickle.load(f, encoding="latin1")
	
	signal = data['signal'][modality]

	if modality == 'wrist':
		signal['Temp'] = signal.pop('TEMP')
		# del signal['TEMP']
	
	baseline_signal = {}
	stress_signal = {}
	
	for k in sampling_freq_dict:

		if sampling_freq_dict[k] == label_samp_freq:
			baseline_labels = data['label'] == 1
			stress_labels = data['label'] == 2
		else:
			labels_new = downsample_labels(data['label'], sampling_freq=sampling_freq_dict[k], label_samp_freq=label_samp_freq)
			baseline_labels = labels_new == 1
			stress_labels = labels_new == 2
		
		baseline_signal[k] = signal[k][baseline_labels].squeeze()
		stress_signal[k] = signal[k][stress_labels].squeeze()

	# baseline_df = pd.DataFrame(baseline_signal)
	# stress_df = pd.DataFrame(stress_signal)

	return baseline_signal, stress_signal


def downsample_labels(labels, sampling_freq, label_samp_freq=700):

	signal_iterator = WI.SingleSignalIterator(signal=labels, window_in_sec=1, shift_in_sec=1, sampling_freq=label_samp_freq)

	labels_new = []
	new_window_width = label_samp_freq//sampling_freq
	t_in = np.arange(label_samp_freq) / label_samp_freq

	idxs = [np.searchsorted(t_in, k/sampling_freq, side='left') for k in range(sampling_freq)] + [label_samp_freq]

	for window in signal_iterator:
		labels_new += [round(np.mean(window[idxs[i]:idxs[i+1]])) for i in range(sampling_freq)]

	return np.array(labels_new)




def get_feature_mat(signal, data_iterator, feature_extractor):
	'''
	This function is used to obtain the feature matrix by constructing the rolling windows.

	Args:
	signal: (pd.DataFrame) It contains the physiological signals. Assumed to atleast have 'EDA', 'ECG', 'Temp' and 'Resp'
	window_size (int): The size of the window (in terms of number of data point/indices in the arrays) 
	shift (int): The size of the shift (in terms of number of data point/indices in the arrays) 

	Returns:

	feature_mat (array-like): The feature matrix corresponding to the input signal.

	'''

	feature_mat = []

	for window in data_iterator:
		feature_mat.append(feature_extractor.get_feature(window))

	return np.array(feature_mat)



def get_train_test_data(test_subjects, modality, window_in_sec, shift_in_sec, feature_dict, sampling_freq_dict, calibration_frac=1):
	'''

	This function is used to generate a train test split.

	Args:
	test_subjects (list): A list of subject indices assumed that are to be assigned the test label (assumed to be within the allowed set)
	window_in_sec (float):  The length of the window (in seconds) to be used to create a single feature vector
	shift_in_sec (float): The time by which the window must be rolled to obtain the next data point.
	sampling_freq  (float): The sampling frequency of the signal in Hz. Assumed to be common across signals.
	calibration_frac (float): The fraction of the baseline data to be used for computing the mean and variance required for calibration

	Returns:
	train_df (pd.DataFrame): A dataframe containing the training dataset
	test_data_list (list): A list of pd.DataFrames each of which correspond to the test data of a given subject


	'''

	all_subjects = set(np.arange(2,18))
	train_subjects = list(all_subjects - set(test_subjects))
	feature_extractor = FE.FeatureExtractor(feature_dict, sampling_freq_dict)

	for i in range(len(train_subjects)):

		if train_subjects[i] == 12:
			continue

		if i == 0:

			train_df = get_subject_features(subject_idx=train_subjects[i], modality=modality, window_in_sec=window_in_sec, shift_in_sec=shift_in_sec, feature_extractor=feature_extractor, sampling_freq_dict=sampling_freq_dict)

		else:

			subject_df = get_subject_features(subject_idx=train_subjects[i], modality=modality, window_in_sec=window_in_sec, shift_in_sec=shift_in_sec, feature_extractor=feature_extractor, sampling_freq_dict=sampling_freq_dict)
			train_df = pd.concat([train_df, subject_df], ignore_index=True)

	test_data_list = []

	for i in test_subjects:

		if i == 12:
			continue

		test_df = get_subject_features(subject_idx=i, modality=modality, window_in_sec=window_in_sec, shift_in_sec=shift_in_sec, feature_extractor=feature_extractor, sampling_freq_dict=sampling_freq_dict, calibration_frac=calibration_frac, include_calibration=False)
		test_data_list.append(test_df)


	return train_df, test_data_list











