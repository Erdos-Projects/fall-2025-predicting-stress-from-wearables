import numpy as np


class SingleSignalIterator():

	def __init__(self, signal, window_in_sec, shift_in_sec, sampling_freq):
		self.signal = signal
		self.window_size = int(round(window_in_sec * sampling_freq))
		self.shift = int(round(shift_in_sec * sampling_freq))
		self.curr_idx = 0
		self.len = len(signal)
		pass

	def __iter__(self):
		return self

	def __next__(self):
		if self.curr_idx + self.window_size > self.len:
			raise StopIteration

		window = self.signal[self.curr_idx:(self.curr_idx+self.window_size)]
		self.curr_idx += self.shift
		return window


class DataIterator():

	def __init__(self, signal, window_in_sec, shift_in_sec, sampling_freq_dict):
		# signal is a pandas dataframe
		self.signal = signal
		self.iterators = {}
		self.window = {}
		for k in sampling_freq_dict:
			try:
				self.iterators[k] = SingleSignalIterator(signal[k], window_in_sec, shift_in_sec, sampling_freq_dict[k])
				self.window[k] = None
			except KeyError:
				print('No signal with name ' + k + ' found')




	def __iter__(self):
		return self

	def __next__(self):
		for k in self.iterators:
			self.window[k] = self.iterators[k].__next__()

		return self.window

