import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# =============================================================================
# Early Detection Metrics
# =============================================================================


def get_early_detection_metrics(y_true_list, y_pred_list, threshold, window_in_sec, shift_in_sec):

	"""
    Compute early detection metrics: false alarms, misdetections, and detection delays.

    This function evaluates how quickly and accurately a model detects a stress onset.
    It uses a sliding window (via convolution) to identify sustained periods of stress
    or non-stress in the ground truth (`y_true`) and predictions (`y_pred`).

    Parameters
    ----------
    y_true_list : list of np.ndarray
        List of ground-truth label arrays for each subject.
        Each array contains binary values (0 = baseline, 1 = stress).
    y_pred_list : list of np.ndarray
        List of predicted label arrays (same shape as y_true_list).
    threshold : int
        Number of consecutive samples required to confirm a stress onset.
    window_in_sec : float
        Duration (in seconds) of the signal window used during feature extraction.
    shift_in_sec : float
        Time shift (in seconds) between consecutive windows.

    Returns
    -------
    false_alarms : list of int
        Count of false alarm events (predicted stress when none present) for each subject.
    misdetections : list of int
        Count of missed detections (true stress not predicted) for each subject.
    detection_delays : list of int
        Estimated detection delays (in seconds) for each subject.
    """

	false_alarms = []
	misdetections = []
	detection_delays = []

	for y_true, y_pred in zip(y_true_list, y_pred_list):

		# Apply convolution to create "sustained activation" signals
		output_true = np.convolve(y_true, np.ones(threshold), mode='valid')
		output_pred = np.convolve(y_pred, np.ones(threshold), mode='valid')

		# Convert convolved outputs to binary "stress present" indicators
		stress_flag = 0

		for i in range(len(output_true)):
			if output_true[i] == 0:
				stress_flag = 0
			elif output_true[i] == threshold:
				stress_flag = 1

			output_true[i] = stress_flag

		stress_flag = 0

		for i in range(len(output_pred)):
			if output_pred[i] == 0:
				stress_flag = 0
			elif output_pred[i] == threshold:
				stress_flag = 1

			output_pred[i] = stress_flag

		# False alarms: predicted stress when none in truth
		# will count when output_true[i] == 0 and output_pred[i] == 1
		fa = np.sum((1 - output_true)*output_pred)

		# Missed detections: true stress not detected  
		# will count when output_true[i] == 1 and output_pred[i] == 0 
		md = np.sum(output_true*(1 - output_pred))  

		# Compute delay: how long until the first detection after stress onset
		delay_idx = 0

		for i in range(len(output_true)):
			if output_true[i] == 0:
				continue
			else:
				if output_pred[i] == 0:
					delay_idx += 1
				else:
					break

		# Detection delay in seconds
		dd = window_in_sec + (threshold - 1 + delay_idx)*shift_in_sec

		false_alarms.append(int(fa))
		misdetections.append(int(md))
		detection_delays.append(int(dd))

	return false_alarms, misdetections, detection_delays

# =============================================================================
# Classification Metrics Over Time
# =============================================================================


def get_classification_metrics(y_true_list, y_pred_list, window_in_sec, shift_in_sec, times):
	"""
    Compute standard classification metrics (precision, recall, F1)
    over time as the prediction horizon increases.

    Parameters
    ----------
    y_true_list : list of np.ndarray
        Ground-truth binary label arrays for each subject.
    y_pred_list : list of np.ndarray
        Predicted binary label arrays for each subject.
    window_in_sec : float
        Duration (in seconds) of the processing window.
    shift_in_sec : float
        Time shift (in seconds) between windows.
    times : list or np.ndarray
        Array of time points (in seconds) at which metrics are evaluated.

    Returns
    -------
    time_arr : list of float
        Time points (subset of `times`) where metrics were computed.
    precision_arr : list of float
        Precision values at each evaluation time.
    recall_arr : list of float
        Recall values at each evaluation time.
    f1_score_arr : list of float
        F1-score values at each evaluation time.
    """

	last_zero_idx = sum(y_true_list[0] == 0) # index before first stress onset
	precision_arr = []
	recall_arr = []
	f1_score_arr = []
	time_arr = []

	for t in times:
		# Check that the time aligns with a valid window position
		if ((t - window_in_sec) >= 0) and ((t - window_in_sec) % shift_in_sec == 0):
			idx = ((t - window_in_sec) // shift_in_sec) + 1

			# Concatenate data from all subjects up to the current time index
			output_true = np.concatenate([y[:(last_zero_idx + idx)] for y in y_true_list])
			output_pred = np.concatenate([y[:(last_zero_idx + idx)] for y in y_pred_list])
			
			# Compute metrics at this time point
			time_arr.append(t)
			precision_arr.append(precision_score(output_true, output_pred))
			recall_arr.append(recall_score(output_true, output_pred))
			f1_score_arr.append(f1_score(output_true, output_pred))

	return time_arr, precision_arr, recall_arr, f1_score_arr


	
 







