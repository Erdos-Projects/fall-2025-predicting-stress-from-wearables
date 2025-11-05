import numpy as np
import neurokit2 as nk
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


# =============================================================================
# Base and Manager Classes
# =============================================================================


class FeatureExtractor():
	"""
    A unified interface for extracting physiological features across multiple signal types.
    
    This class dynamically initializes signal-specific processors (e.g., ECG, EDA, Resp)
    and aggregates their extracted features into a single NumPy array.
    
    Parameters
    ----------
    feature_dict : dict
        Dictionary specifying which features to extract for each signal type.
        Example: {'EDA': {'SCL': ['mean'], 'SCR': ['std']}, 'ECG': ['mean_rate']}
    sampling_freq_dict : dict
        Dictionary mapping each signal type to its sampling frequency.
    """

	def __init__(self, feature_dict, sampling_freq_dict):
		self.feature_dict = feature_dict
		self.sampling_freq_dict = sampling_freq_dict

		# Mapping from signal type to its feature processor constructor
		self.obj_initializer = { 'EDA' : lambda x: EDAFeatureProcessor(x),
								 'ECG' : lambda x: ECGFeatureProcessor(x),
								 'Temp' : lambda x: TempFeatureProcessor(x),
								 'Resp' : lambda x: RespFeatureProcessor(x),
								 'EMG' : lambda x: EMGFeatureProcessor(x),
								 'BVP' : lambda x: BVPFeatureProcessor(x)
								}
		self.set_feature_extractor()

	def set_feature_extractor(self):
		"""Initialize all feature processor instances and collect their feature names."""

		feature_extractor = {}
		feature_name_list = []

		for k in self.feature_dict:
			fp = self.obj_initializer[k](self.sampling_freq_dict[k])
			fp.set_feature_extractor(self.feature_dict[k])
			feature_name_list += fp.feature_names
			feature_extractor[k] = fp

		self.feature_extractor = feature_extractor
		self.feature_name_list = feature_name_list

	def get_feature(self, signal):
		"""
        Extract features for each signal type and concatenate results.
        
        Parameters
        ----------
        signal : dict
            Dictionary mapping signal types to raw arrays (e.g., {'EDA': np.array(...)}).
        
        Returns
        -------
        np.ndarray
            Concatenated feature vector for all signal types.
        """

		feature = []

		for k in self.feature_extractor:
			feature += self.feature_extractor[k].get_features(signal[k])

		return np.array(feature)


# =============================================================================
# Base Processor Class
# =============================================================================

class FeatureProcessor():
	"""
    Base class for all signal-specific feature processors.
    
    Provides core utilities such as slope calculation, inter-peak timing,
    and mapping feature names to computation functions.
    """

	def __init__(self, sampling_freq):
		self.sampling_freq = sampling_freq
		self._set_func_mapper()

	def clean_signal(self, signal):
		"""To be implemented by subclasses — should return preprocessed signal(s)."""
		pass

	def get_inter_peak_times(self, peaks):
		"""Return inter-peak intervals (ms) given a binary peak vector."""
		return (np.diff(np.where(peaks)[0]/self.sampling_freq))*1000 # in ms

	def _slope(self, signal):
		"""Compute linear trend slope (via simple regression)."""
		t = np.arange(len(signal)) / self.sampling_freq
		model = LinearRegression().fit(t.reshape(-1, 1), signal)

		return model.coef_[0]

	def _set_func_mapper(self):
		"""Define available feature computation functions."""

		func_map = {}

		func_map['mean'] = lambda x: np.mean(x[0][~np.isnan(x[0])])
		func_map['std'] = lambda x: np.std(x[0][~np.isnan(x[0])])
		func_map['min'] = lambda x: np.min(x[0][~np.isnan(x[0])])
		func_map['max'] = lambda x: np.max(x[0][~np.isnan(x[0])])
		func_map['slope'] = lambda x: self._slope(x[0][~np.isnan(x[0])])

		func_map['mean_rate'] = lambda x: np.mean(60000/x[1])
		func_map['std_rate'] = lambda x: np.std(60000/x[1])
		func_map['inter_peak_time'] = lambda x: np.mean(x[1])
		func_map['SDNN'] = lambda x: np.std(x[1])
		func_map['counts'] = lambda x: len(x[1])

		func_map['RMSSD'] = lambda x: np.sqrt(np.mean(np.diff(x[1])**2))

		func_map['HF'] = lambda x: x[2]['HRV_HF'].values[0]
		func_map['VHF'] = lambda x: x[2]['HRV_VHF'].values[0]
		func_map['TP'] = lambda x: x[2]['HRV_TP'].values[0]
		func_map['HF_norm'] = lambda x: x[2]['HRV_HFn'].values[0]

		self.func_map =  func_map

	def _set_feature_extractor(self, feature_list):
		"""Return callable to compute all selected features from a cleaned signal tuple."""

		feature_extractor = lambda x: [self.func_map[f](x) for f in feature_list]

		return feature_extractor



# =============================================================================
# Signal-Specific Feature Processor Classes
# =============================================================================


class EDAFeatureProcessor(FeatureProcessor):
	"""Feature extractor for Electrodermal Activity (EDA)."""

	def __init__(self, sampling_freq):
		super().__init__(sampling_freq)

	def clean_signal(self, signal):

		# Process and clean the signal to obtain the various components
		eda_signal, _ = nk.eda_process(signal, sampling_freq=self.sampling_freq)

		# Extract the SCL and SCR signal separately along with SCR peaks
		scl_signal = (eda_signal.EDA_Tonic.values, None)
		scr_signal = (eda_signal.SCR_Amplitude.values, self.get_inter_peak_times(eda_signal.SCR_Peaks.values))

		return scl_signal, scr_signal

	def set_feature_extractor(self, feature_dict):

		self.feature_extractor = {'SCL' : self._set_feature_extractor(feature_dict['SCL']), 
								  'SCR': self._set_feature_extractor(feature_dict['SCR'])}
		self.feature_names = ['SCL_' + x for x in  feature_dict['SCL']] + ['SCR_' + x for x in  feature_dict['SCR']]


	def get_features(self, signal):

		scl_signal, scr_signal = self.clean_signal(signal)

		return self.feature_extractor['SCL'](scl_signal) + self.feature_extractor['SCR'](scr_signal)



class TempFeatureProcessor(FeatureProcessor):
	"""Feature extractor for Temperature signals."""

	def __init__(self, sampling_freq):
		super().__init__(sampling_freq)

	def clean_signal(self, signal):

		# Temperature signal is usually clean and needs no additional processing
		return (signal, None)

	def set_feature_extractor(self, feature_list):

		self.feature_extractor = self._set_feature_extractor(feature_list)
		self.feature_names = ['Temp_' + x for x in feature_list]


	def get_features(self, signal):

		temp_signal = self.clean_signal(signal)

		return self.feature_extractor(temp_signal) 



class ECGFeatureProcessor(FeatureProcessor):
	"""Feature extractor for Electrocardiogram (ECG) signals."""

	def __init__(self, sampling_freq):
		super().__init__(sampling_freq)

	def clean_signal(self, signal):

		# Process and clean the signal for further processing
		ecg_cleaned = nk.ecg_clean(signal, sampling_rate=self.sampling_freq)

		# Obtain the peaks from the signal 
		peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.sampling_freq, correct_artifacts=True)

		# Use the peaks to extract frequency domain information
		hrv_freq = nk.hrv_frequency(peaks, sampling_rate=self.sampling_freq)

		return (ecg_cleaned, self.get_inter_peak_times(peaks['ECG_R_Peaks'].values), hrv_freq)

	def set_feature_extractor(self, feature_list):

		self.feature_extractor = self._set_feature_extractor(feature_list)
		self.feature_names = ['ECG_' + x for x in feature_list]

	def get_features(self, signal):

		ecg_signal = self.clean_signal(signal)

		return self.feature_extractor(ecg_signal) 


class RespFeatureProcessor(FeatureProcessor):
	"""Feature extractor for Respiration (RSP) signals."""

	def __init__(self, sampling_freq):
		super().__init__(sampling_freq)

	def clean_signal(self, signal):

		# Process and clean the signal for further processing
		rsp_signal, _ = nk.rsp_process(signal, sampling_freq=self.sampling_freq)

		# Seperate the signal amplitude and RVT components
		rsp_amp = (rsp_signal.RSP_Amplitude.values, self.get_inter_peak_times(rsp_signal.RSP_Peaks.values))
		rsp_rvt = (rsp_signal.RSP_RVT.values, None)


		return rsp_amp, rsp_rvt

	def set_feature_extractor(self, feature_dict):

		self.feature_extractor = {'Amp' : self._set_feature_extractor(feature_dict['Amp']), 
								  'RVT': self._set_feature_extractor(feature_dict['RVT'])}
		self.feature_names = ['Resp_Amp_' + x for x in feature_dict['Amp']] + ['Resp_RVT_' + x for x in feature_dict['RVT']]

	def get_features(self, signal):

		rsp_amp, rsp_rvt = self.clean_signal(signal)

		return self.feature_extractor['Amp'](rsp_amp) + self.feature_extractor['RVT'](rsp_rvt)


class EMGFeatureProcessor(FeatureProcessor):
	"""Feature extractor for Electromyography (EMG) signals."""

	def __init__(self, sampling_freq):
		super().__init__(sampling_freq)

	def clean_signal(self, signal):

		# Clean the signal
		emg_cleaned, _ = nk.emg_process(signal, sampling_rate=self.sampling_freq)

		return (emg_cleaned.EMG_Amplitude.values, self.get_inter_peak_times(emg_cleaned.EMG_Onsets.values))

	def set_feature_extractor(self, feature_list):

		self.feature_extractor = self._set_feature_extractor(feature_list)
		self.feature_names = ['EMG_' + x for x in feature_list]

	def get_features(self, signal):

		emg_signal = self.clean_signal(signal)

		return self.feature_extractor(emg_signal) 


class BVPFeatureProcessor(FeatureProcessor):
	"""Feature extractor for Blood Volume Pulse (BVP/PPG) signals."""

	def __init__(self, sampling_freq):
		super().__init__(sampling_freq)

	def clean_signal(self, signal):

		# Filter the signal to remove noise and clean it
		bvp_signal = nk.signal_filter(signal, lowcut=0.5, highcut=8, sampling_rate=self.sampling_freq, method='butterworth')

		# Extract peaks from the signal
		peaks, _ = nk.ppg_peaks(bvp_signal, sampling_rate=self.sampling_freq)

		return (bvp_signal, self.get_inter_peak_times(peaks.PPG_Peaks.values))

	def set_feature_extractor(self, feature_list):

		self.feature_extractor = self._set_feature_extractor(feature_list)
		self.feature_names = ['BVP_' + x for x in feature_list]

	def get_features(self, signal):

		bvp_signal = self.clean_signal(signal)

		return self.feature_extractor(bvp_signal) 




