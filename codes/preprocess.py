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
	modality (str): 'chest' or 'wrist' specfiying which modality data has to be taken from
	window_in_sec (float):  The length of the window (in seconds) to be used to create a single feature vector
	shift_in_sec (float): The time by which the window must be rolled to obtain the next data point.
	feature_extractor (FeatureExtractor): An object of the FeatureExtractor class that can be used to obtain feature for given signal
	sampling_freq_dict  (dictionary): The sampling frequencies of the different signals in Hz
	calibration_frac (float): The fraction of the baseline data to be used for computing the mean and variance required for calibration
	include_calibration (bool): Denotes where to include the data used for calibration in the generated features

	Returns:

	dataset (pd.DataFrame): The dataset corresponding to the subject consisting of baseline data (label = 0) followed by 
							the stress data (label = 1). This contains all the windows arranged chronologically.


	'''
	
	# Obtain the separate baseline and stress signal for a given modality
	baseline_df, stress_df = get_separated_signals(subject_idx, modality, sampling_freq_dict) # NOTE: Updated to have baseline_df and stress_df as dictionaries instead of dataframes

	# Data iterators for baseline and stress data to obtain the rolling windows
	baseline_iterator = WI.DataIterator(baseline_df, window_in_sec, shift_in_sec, sampling_freq_dict)
	stress_iterator = WI.DataIterator(stress_df, window_in_sec, shift_in_sec, sampling_freq_dict)

	# Obtain the feature matrix
	baseline_feature_mat = get_feature_mat(baseline_df, baseline_iterator, feature_extractor)
	stress_feature_mat = get_feature_mat(stress_df, stress_iterator, feature_extractor)

	# Set the maximum number of indices to be used for calibration
	max_calibration_idx = int(round(calibration_frac*baseline_feature_mat.shape[0]))

	# Obtain the mean and standard deviation for calibration
	baseline_mean = baseline_feature_mat[:max_calibration_idx, :].mean(axis=0)
	baseline_std = baseline_feature_mat[:max_calibration_idx, :].std(axis=0)

	# Normalize the baseline matrix
	baseline_feature_mat_normalized = baseline_feature_mat - np.tile(baseline_mean, [baseline_feature_mat.shape[0], 1])
	baseline_feature_mat_normalized = baseline_feature_mat_normalized/np.tile(baseline_std, [baseline_feature_mat.shape[0], 1])

	# Normalize the stree feature matrix. Note we use the baseline normalization here.
	stress_feature_mat_normalized = stress_feature_mat - np.tile(baseline_mean, [stress_feature_mat.shape[0], 1])
	stresss_feature_mat_normalized = stress_feature_mat_normalized/np.tile(baseline_std, [stress_feature_mat.shape[0], 1])

	# Check if the data used for calibration has to be included in the dataset or not and accordingly make the dataset.
	if include_calibration:
		baseline_normalized_df = pd.DataFrame(baseline_feature_mat_normalized, columns=feature_extractor.feature_name_list)
	else:
		baseline_normalized_df = pd.DataFrame(baseline_feature_mat_normalized[max_calibration_idx:, :], columns=feature_extractor.feature_name_list)
	stress_normalized_df = pd.DataFrame(stress_feature_mat_normalized, columns=feature_extractor.feature_name_list)

	# Add labels
	baseline_normalized_df['label'] = 0
	stress_normalized_df['label'] = 1

	# Add subject index
	baseline_normalized_df['subject_idx'] = subject_idx
	stress_normalized_df['subject_idx'] = subject_idx


	return pd.concat([baseline_normalized_df, stress_normalized_df], ignore_index=True)


def get_separated_signals(subject_idx, modality, sampling_freq_dict, label_samp_freq=700):
	'''

	A function to obtain the seperated baseline and stress signals from the original signal.

	Args:
	subject_idx (int): The index of the subject. Has to be in the set {2,3,4,5,6,7,8,9,10,11,13,14,15,16,17}.
	modality (str): 'chest' or 'wrist' specfiying which modality data has to be taken from
	sampling_freq_dict  (dictionary): The sampling frequencies of the different signals in 
	label_samp_freq (int): The sampling frequency (in Hz) of the label array
	
	Returns:
	baseline_signal (dictionary) : A dictionary containing the signals corresponding to the baseline period
	stress_signal (dictionary) : A dictionary containing the signals corresponding to the stress period

	'''

	# Read the data
	with open(f'../WESAD/S{subject_idx}/S{subject_idx}.pkl', 'rb') as f:
		data = pickle.load(f, encoding="latin1")
	
	# Obtain the signal of the relevant modality
	signal = data['signal'][modality]

	# Rename keys for consistency
	if modality == 'wrist':
		signal['Temp'] = signal.pop('TEMP')
	
	baseline_signal = {}
	stress_signal = {}
	
	for k in sampling_freq_dict:

		if sampling_freq_dict[k] == label_samp_freq:
			baseline_labels = data['label'] == 1
			stress_labels = data['label'] == 2
		else:
			# Downsample the labels if the sampling frequency of signal is smaller than that of labels
			labels_new = downsample_labels(data['label'], sampling_freq=sampling_freq_dict[k], label_samp_freq=label_samp_freq)
			baseline_labels = labels_new == 1
			stress_labels = labels_new == 2
		
		baseline_signal[k] = signal[k][baseline_labels].squeeze()
		stress_signal[k] = signal[k][stress_labels].squeeze()

	return baseline_signal, stress_signal


def downsample_labels(labels, sampling_freq, label_samp_freq=700):
	'''

	A function used to downsample the label array to a given sampling frequency

	Args:

	labels (array of ints): An array of integers containing the different labels sampled at label_samp_freq
	sampling_freq (int): The required sampling frequency in Hz
	label_samp_freq (int): The original sampling frequency of the labels array

	Returns:

	labels_new (array of ints): A array of labels with samples at the sampling_freq
	'''

	# Data iterator for the label signal. We will iterate in windows of 1 sec.
	signal_iterator = WI.SingleSignalIterator(signal=labels, window_in_sec=1, shift_in_sec=1, sampling_freq=label_samp_freq)


	labels_new = []
	t_in = np.arange(label_samp_freq) / label_samp_freq

	# Create the list of indices that will serve at the end point of the new intervals
	idxs = [np.searchsorted(t_in, k/sampling_freq, side='left') for k in range(sampling_freq)] + [label_samp_freq]

	for window in signal_iterator:
		# Average and round off the labels in a given sub-interval to obtain the label for that sub-interval
		labels_new += [round(np.mean(window[idxs[i]:idxs[i+1]])) for i in range(sampling_freq)]

	return np.array(labels_new)




def get_feature_mat(signal, data_iterator, feature_extractor):
	'''
	This function is used to obtain the feature matrix by constructing the rolling windows.

	Args:
	signal: (dictionary) Dictionary contains the different physiological signals. Assumed to have valid keys.
	data_iterator (DataIterator): An object of class DataIterator used to iterate over the different window in the dataset
	feature_extractor (FeatureExtractor): An object of the FeatureExtractor class that can be used to obtain feature for given signal

	Returns:

	feature_mat (array-like): The feature matrix corresponding to the input signal.


	'''

	feature_mat = []

	# Iterate over different windows and extract the features
	for window in data_iterator:
		feature_mat.append(feature_extractor.get_feature(window))

	return np.array(feature_mat)



def get_train_test_data(test_subjects, modality, window_in_sec, shift_in_sec, feature_dict, sampling_freq_dict, calibration_frac=1):
	'''

	This function is used to generate a train test split.

	Args:
	test_subjects (list): A list of subject indices assumed that are to be assigned the test label (assumed to be within the allowed set)
	modality (str): 'chest' or 'wrist' specfiying which modality data has to be taken from
	window_in_sec (float):  The length of the window (in seconds) to be used to create a single feature vector
	shift_in_sec (float): The time by which the window must be rolled to obtain the next data point.
	feature_dict (dictionary): A dictionary specifying the different features required for different signals
	sampling_freq_dict  (dictionary): The sampling frequencies of the different signals in Hz
	calibration_frac (float): The fraction of the baseline data to be used for computing the mean and variance required for calibration

	Returns:
	train_df (pd.DataFrame): A dataframe containing the training dataset
	test_df (pd.DataFrame): A dataframe containing the test dataset

	'''

	all_subjects = set(np.arange(2,18))
	train_subjects = list(all_subjects - set(test_subjects))
	feature_extractor = FE.FeatureExtractor(feature_dict, sampling_freq_dict)

	for i in range(len(train_subjects)):

		# Skip 12 since that is not in the dataset
		if train_subjects[i] == 12:
			print(train_subjects[i])
			continue

		# Obtain the feature matrix for the given subject
		if i == 0:
			train_df = get_subject_features(subject_idx=train_subjects[i], modality=modality, window_in_sec=window_in_sec, shift_in_sec=shift_in_sec, feature_extractor=feature_extractor, sampling_freq_dict=sampling_freq_dict)

		else:
			subject_df = get_subject_features(subject_idx=train_subjects[i], modality=modality, window_in_sec=window_in_sec, shift_in_sec=shift_in_sec, feature_extractor=feature_extractor, sampling_freq_dict=sampling_freq_dict)
			train_df = pd.concat([train_df, subject_df], ignore_index=True)

		print(train_subjects[i])


	for i, test_idx in enumerate(test_subjects):

		# Skip 12 since that is not in the dataset
		if test_idx == 12:
			continue

		# Obtain the feature matrix for the given subject
		if i == 0:
			test_df = get_subject_features(subject_idx=test_idx, modality=modality, window_in_sec=window_in_sec, shift_in_sec=shift_in_sec, feature_extractor=feature_extractor, sampling_freq_dict=sampling_freq_dict, calibration_frac=calibration_frac, include_calibration=False)

		else:
			subject_df = get_subject_features(subject_idx=test_idx, modality=modality, window_in_sec=window_in_sec, shift_in_sec=shift_in_sec, feature_extractor=feature_extractor, sampling_freq_dict=sampling_freq_dict, calibration_frac=calibration_frac, include_calibration=False)
			test_df = pd.concat([test_df, subject_df], ignore_index=True)

		print(test_idx)



	return train_df, test_df 











