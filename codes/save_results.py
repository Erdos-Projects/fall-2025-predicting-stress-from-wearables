from performance_analysis import *

window_shift_pairs = [(10,5), (10,10), (20,5), (20,10), (20, 20), (30, 5), (30, 10), (30, 15), (30,30), (60, 5), (60, 15), (60, 30), (60, 60)]
modality = 'chest'
# modality = 'wrist'
methods = ['LR', 'RF', 'LDA', 'XG']
method_names = ['Logistic Regression', 'Random Forest', 'Linear Discriminant Analysis', 'XGBoost']
times = [30, 35, 40, 45, 60, 75, 90, 105, 120, 150, 180, 300, 600]

print('\n Starting with small features...... \n')

for w,s in window_shift_pairs:

	try:
		train_df, test_data_list = load_train_test_data(modality=modality, window_in_sec=w, shift_in_sec=s)
	except FileNotFoundError:
		continue

	filename = f'../results/{modality}_{w}_{s}.txt'

	with open(filename, 'w') as f:

		for method, method_name in zip(methods, method_names):

			model, feature_cols = get_predictor(train_df=train_df, method=method)
			y_true_list, y_pred_list = get_predictions(test_data_list=test_data_list, model=model, feature_cols=feature_cols)

			f.write('Method: ' + method_name + '\n\n')

			for threshold in  [1,2,3,4,5]:
				false_alarms, misdetections, detection_delays = get_early_detection_metrics(y_true_list=y_true_list, y_pred_list=y_pred_list, threshold=threshold, window_in_sec=w, shift_in_sec=s)
				f.write(f'For threshold = {threshold}:\n')
				f.write('False Alarms:' + ' '.join([str(fa) for fa in false_alarms]) + '\n')
				f.write('Miss Detections:' + ' '.join([str(md) for md in misdetections]) + '\n')
				f.write('Detection Delays (in sec):' + ' '.join([str(dd) for dd in detection_delays]) + '\n')

			f.write('\n')
			f.write('Classification performance: \n')

			time_arr, precision_arr, recall_arr, f1_score_arr = get_classification_metrics(y_true_list=y_true_list, y_pred_list=y_pred_list, window_in_sec=w, shift_in_sec=s, times=times)

			f.write('Time stamps:' + ' '.join([str(t) for t in time_arr]) + '\n')
			f.write('Precision scores:' + ' '.join([str(ps) for ps in precision_arr]) + '\n')
			f.write('Recall scores:' + ' '.join([str(rs) for rs in recall_arr]) + '\n')
			f.write('F1 scores:' + ' '.join([str(f1s) for f1s in f1_score_arr]) + '\n')

			f.write('------------------------------------------- \n\n')

	print(f'({w},{s})', ' Done')

print('\n Trying all features now...... \n')

for w,s in window_shift_pairs:

	try:
		train_df, test_data_list = load_train_test_data(modality=modality, window_in_sec=w, shift_in_sec=s, all_features=True)
	except FileNotFoundError:
		continue

	filename = f'../results/{modality}_{w}_{s}_all.txt'

	with open(filename, 'w') as f:

		for method, method_name in zip(methods, method_names):

			model, feature_cols = get_predictor(train_df=train_df, method=method)
			y_true_list, y_pred_list = get_predictions(test_data_list=test_data_list, model=model, feature_cols=feature_cols)

			f.write('Method: ' + method_name + '\n\n')

			for threshold in  [1,2,3,4,5]:
				false_alarms, misdetections, detection_delays = get_early_detection_metrics(y_true_list=y_true_list, y_pred_list=y_pred_list, threshold=threshold, window_in_sec=w, shift_in_sec=s)
				f.write(f'For threshold = {threshold}:\n')
				f.write('False Alarms:' + ' '.join([str(fa) for fa in false_alarms]) + '\n')
				f.write('Miss Detections:' + ' '.join([str(md) for md in misdetections]) + '\n')
				f.write('Detection Delays (in sec):' + ' '.join([str(dd) for dd in detection_delays]) + '\n')

			f.write('\n')
			f.write('Classification performance: \n')

			time_arr, precision_arr, recall_arr, f1_score_arr = get_classification_metrics(y_true_list=y_true_list, y_pred_list=y_pred_list, window_in_sec=w, shift_in_sec=s, times=times)

			f.write('Time stamps:' + ' '.join([str(t) for t in time_arr]) + '\n')
			f.write('Precision scores:' + ' '.join([str(ps) for ps in precision_arr]) + '\n')
			f.write('Recall scores:' + ' '.join([str(rs) for rs in recall_arr]) + '\n')
			f.write('F1 scores:' + ' '.join([str(f1s) for f1s in f1_score_arr]) + '\n')

			f.write('------------------------------------------- \n\n')

	print(f'({w},{s})', ' Done')


