# =============================================================================
# Single-Signal Iterator
# =============================================================================


class SingleSignalIterator():
	"""
    Iterator that yields fixed-length windows from a single continuous signal.

    Parameters
    ----------
    signal : array-like
        One-dimensional signal data (e.g., from EDA, ECG, etc.).
    window_in_sec : float
        Window duration in seconds.
    shift_in_sec : float
        Step size (shift) between consecutive windows in seconds.
    sampling_freq : float
        Sampling frequency (Hz) of the signal.

    Attributes
    ----------
    window_size : int
        Number of samples per window.
    shift : int
        Number of samples between consecutive windows.
    curr_idx : int
        Current index of the sliding window start.
    """

	def __init__(self, signal, window_in_sec, shift_in_sec, sampling_freq):
		self.signal = signal
		self.window_size = int(round(window_in_sec * sampling_freq))
		self.shift = int(round(shift_in_sec * sampling_freq))
		self.curr_idx = 0
		self.len = len(signal)
		pass

	def __iter__(self):
		"""Return iterator object (self)."""
		return self

	def __next__(self):
		"""
        Return the next signal window.

        Raises
        ------
        StopIteration
            When there are no more complete windows left in the signal.
        """
		if self.curr_idx + self.window_size > self.len:
			raise StopIteration

		# Extract current window slice
		window = self.signal[self.curr_idx:(self.curr_idx+self.window_size)]

		# Advance window position
		self.curr_idx += self.shift

		return window


# =============================================================================
# Multi-Signal Iterator
# =============================================================================


class DataIterator():
	"""
    Iterator that synchronously yields windows from multiple signals.

    Parameters
    ----------
    signal : pd.DataFrame
        A DataFrame where each column represents a different signal type
        (e.g., EDA, ECG, Temp, Resp, etc.).
    window_in_sec : float
        Window duration in seconds.
    shift_in_sec : float
        Step size between consecutive windows in seconds.
    sampling_freq_dict : dict
        Mapping of signal names to their sampling frequencies.

    Attributes
    ----------
    iterators : dict
        Dictionary of SingleSignalIterator objects for each signal.
    window : dict
        Dictionary holding the most recent windowed data for each signal.
    """

	def __init__(self, signal, window_in_sec, shift_in_sec, sampling_freq_dict):
		# signal is a pandas dataframe
		self.signal = signal
		self.iterators = {}
		self.window = {}

		# Create individual iterators for all available signals
		for k in sampling_freq_dict:
			try:
				self.iterators[k] = SingleSignalIterator(signal[k], window_in_sec, shift_in_sec, sampling_freq_dict[k])
				self.window[k] = None
			except KeyError:
				print('No signal with name ' + k + ' found')




	def __iter__(self):
		"""Return iterator object (self)."""
		return self

	def __next__(self):
		"""
        Return a dictionary containing the next window for each signal.

        Raises
        ------
        StopIteration
            When any of the underlying signal iterators run out of data.
        """
        # Fetch next window for each available signal
		for k in self.iterators:
			self.window[k] = self.iterators[k].__next__()

		return self.window

