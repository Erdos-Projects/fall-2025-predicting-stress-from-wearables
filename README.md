# Early Detection of Stress Episodes from Wearables

Shijun Sun, Raagini Patki, Christiana Mavroyiakoumou, Alessandro Podo

We combine multimodal physiological signals such as ECG, skin temperature, skin conductance, and respiration with the objective of detecting (acute) stress episodes as early as possible after they are triggered.

<!-- to forecast whether someone is likely to experience a stress episode in the next 5–30 minutes. -->

## Dataset: 
public dataset WESAD https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html and accompanying paper https://ubi29.informatik.uni-siegen.de/usi/pdf/ubi_icmi2018.pdf.

## Method & key results:
1. Extract features from raw multimodal signals in the WESAD dataset using a sliding window approach. Calibrate features for each subject and do a test-train split.  
2. Train models with extracted features, exploring classical ML methods: Logistic Regression, Random Forests, Linear Discriminant Analysis (LDA), XGBoost.
3. Use our best-performing trained models (i.e. using Random Forests and LDA) to perform robust early detection of stress within 40 sec (90 sec) after stress is triggered, using chest (wrist) data from wearables. 
<!-- 2. Explore deep time series approaches such as 1D CNNs -->

## Why is this research-worthy:
1. Existing research has largely focused on classifying b/w stressed and relaxed states of subjects using physiological signals from wearables. Here, we train models to accurately and robustly detect stress within 40 sec of its onset. Our project targets this research gap.
2. In real-life scenarios, early detection of stress allows for early interventions, which can guide appropriate actions to prevent prolonged/escalated stress responses. This can help support increased well-being and productivity among the wearable devices' users. 

## References:
1. Review: https://pmc.ncbi.nlm.nih.gov/articles/PMC11230864/
2. Another review: https://arxiv.org/pdf/2209.15137
3. Review of biomarkers related to stress. https://onlinelibrary.wiley.com/doi/10.1002/dev.22490
4. ML study that predicts stress up to 1 min ahead: https://arxiv.org/abs/2106.07542
5. Relevant. See their Table 4 for a benchmark of WESAD data using different models. https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control
6. ML time series for stress duration prediction (predicts the duration of the next stress episode from a prior stress episode). Not directly related, but their method may be useful for us. https://dl.acm.org/doi/10.1145/2858036.2858218
7. For Personalization - https://arxiv.org/pdf/2107.05666 (Used WESAD)

--------------------------------------------

## Navigating the Repository

The repo has three main folders, codes, datasets and results.  

**Datasets**  

This has all the datasets that we have extracted from the raw signals. Each file is named of the form a_b_c_d.csv. Here 'a' denotes whether it is training or test and 'b' denotes if the data corresponds to chest or wrist. The numbers c and d denote window size (in seconds) and the shift between windows (in seconds) while extracting the features from raw signal using rolling window. Some datasets all have 'all' at the end, which denotes the datasets with all possible features. The rest of them have a subset of the features.

**Codes**  

This folder contains all the codes. If you want to obtain the results in the as shown in the results folder, run save_results.py. It will default save the results corresponding to 'chest' datasets. For 'wrist' results, just uncomment line 5 (and comment line 4) and rerun the code. The other codes are well-commented and should be easy to follow. The codes require the folder WESAD to run which contains the raw signals. This is not available in the repository but can be downloaded via the link mentioned in earlier in the file. 

**Results**  

This folder contains the results of all our tests on the datasets will different window sizes, shifts, models, modalities, thresholds, and feature combinations. Each file contains the performance on a given dataset and reports the number of false alarms, misdetections, detection delays on all test subjects with different thresholds as part of early detection. For the classification task, precision, recall, f1 score and accuracy are reported at different times after the stress was induced. All these results are evaluated for all the four models in each file.


------------------------------------------------

## Additional Datasets for Future Directions 

**SWELL:** 

25 people performed typical knowledge work. A varied set of data was recorded: computer logging, facial expression from camera recordings, body postures from a Kinect 3D sensor and heart rate (variability) and skin conductance from body sensors. Dataset not only contains raw data, but also preprocessed data and extracted features. The participants' stress was assessed with validated questionnaires
https://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html 

**Exam Stress:**

data contains electrodermal activity, heart rate, blood volume pulse, skin surface temperature, inter beat interval and accelerometer data recorded during three exam sessions https://physionet.org/content/wearable-exam-stress/1.0.0/

**BIOSTRESS**
https://www.kaggle.com/datasets/orvile/biostress-dataset

### For stress vs exercise:
https://www.nature.com/articles/s41597-025-04845-9 and Dataset: https://physionet.org/content/wearable-device-dataset/1.0.0/



