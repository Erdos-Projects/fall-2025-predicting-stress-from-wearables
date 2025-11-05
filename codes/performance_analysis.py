import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from metrics import *

# =============================================================================
# Data Loading Utilities
# =============================================================================


def load_train_test_data(modality, window_in_sec, shift_in_sec, all_features=False):
	"""
    Load and preprocess train/test data for a given physiological modality.

    Parameters
    ----------
    modality : str
        The signal modality --- 'chest' or 'wrist'.
    window_in_sec : int or float
        Window length (in seconds) used for feature extraction.
    shift_in_sec : int or float
        Shift length (in seconds) between consecutive windows.

    Returns
    -------
    train_df : pd.DataFrame
        Cleaned training dataframe with NaNs filled and empty columns dropped.
    test_data_list : list of pd.DataFrame
        List of test dataframes, one per unique subject, aligned to train_df columns.
    """

	if all_features:
		train_df = pd.read_csv(f'train_{modality}_{window_in_sec}_{shift_in_sec}_all.csv')
		test_df = pd.read_csv(f'test_{modality}_{window_in_sec}_{shift_in_sec}_all.csv')
	else:
		train_df = pd.read_csv(f'train_{modality}_{window_in_sec}_{shift_in_sec}.csv')
		test_df = pd.read_csv(f'test_{modality}_{window_in_sec}_{shift_in_sec}.csv')

	if 'subject_idx' not in train_df.columns:
		test_df.rename(columns={'subject':'subject_idx'}, inplace=True)
		train_df.rename(columns={'subject':'subject_idx'}, inplace=True)

	# Drop columns entirely NaN, then fill remaining NaNs with zeros
	train_df = train_df.dropna(axis=1, how='all')
	train_df = train_df.fillna(0.0)

	# Split test data by subject for subject-level evaluation

	if 'subject_idx' not in test_df.columns:
		test_df.rename(columns={'subject':'subject_idx'}, inplace=True)

	test_subjects = test_df['subject_idx'].unique()

	test_data_list = []

	for idx in test_subjects:
		test = test_df[test_df['subject_idx'] == idx]
		# Align columns with training data (important if test lacks some features)
		test = test[train_df.columns].fillna(0.0)
		test_data_list.append(test)

	return train_df, test_data_list


# =============================================================================
# Model Training and Prediction
# =============================================================================

def get_predictor(train_df, method):

	"""
    Train a classifier on the provided training data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data including feature columns and 'label'/'subject_idx' columns.
    method : str
        Type of classifier to train:
        'LR' = Logistic Regression
        'RF' = Random Forest
        'LDA' = Linear Discriminant Analysis
        'XG' = XGBoost

    Returns
    -------
    model : sklearn or xgboost model
        Trained model instance.
    feature_cols : list of str
        Names of feature columns used during training.
    """

    # Exclude label and subject index columns from features
	feature_cols = [c for c in train_df.columns if c not in ['label', 'subject_idx', 'subject', 'win_start_sec']]
	X_train = train_df[feature_cols].values
	y_train = train_df['label'].values

	# Select classifier type
	if method == 'LR':
		model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=500, C=10.0)
	elif method == 'RF':
		model = RandomForestClassifier(n_estimators=7, class_weight='balanced', random_state=42)
	elif method == 'LDA':
		model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
	elif method == 'XG':
		model = XGBClassifier(learning_rate=0.05,max_depth=5,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=42,objective='binary:logistic')
	else:
		raise ValueError(f"Unsupported method: {method}")

	# Fit model
	model.fit(X_train, y_train)

	return model, feature_cols

def get_predictions(test_data_list, model, feature_cols):
	"""
    Generate predictions for each test subject using a trained model.

    Parameters
    ----------
    test_data_list : list of pd.DataFrame
        List of test dataframes, one per subject.
    model : fitted model
        Trained classifier (sklearn or xgboost).
    feature_cols : list of str
        Feature columns to use for prediction.

    Returns
    -------
    y_true_list : list of np.ndarray
        Ground-truth label arrays for each subject.
    y_pred_list : list of np.ndarray
        Predicted label arrays for each subject.
    """

	y_true_list = []
	y_pred_list = []

	for test_df in test_data_list:
		# Predict class labels (optionally, probabilities can be added)
		y_pred = model.predict(test_df[feature_cols].values)
		y_true = test_df['label'].values
		# y_prob = model.predict_proba(test_df[feature_cols].values)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(test_df[feature_cols].values)

		y_true_list.append(y_true)
		y_pred_list.append(y_pred)

	return y_true_list, y_pred_list






