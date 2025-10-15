# Predicting Stress Episodes from Wearables

Shijun Sun, Raagini Patki, Christiana Mavroyiakoumou, Alessandro Podo

The goal is to combine signals such as heart rate, heart rate variability, skin temperature, skin conductance, and motion to forecast whether someone is likely to experience a stress episode in the next 5–30 minutes.

## Dataset: 
public dataset WESAD https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

## Method:
1. Build baseline models using classical ML methods (e.g., logistic regression, random forests)
2. Explore deep time series approaches such as 1D CNNs

## Why is this research-worthy:
1. Predicting stress is possible: stress responses involve gradual changes in HRV, EDA, temperature, and respiration before conscious awareness, making short-term forecasting possible.
2. Existing research shows strong real-time detection of stress and ~1-minute lead-time prediction in controlled scenarios, but there is little work rigorously quantifying 5–30 minute early warnings, and robustness to missing/noisy sensors. This project targets this research gap.
3. High impact and relatable to general audience

## References:
1. Review: https://pmc.ncbi.nlm.nih.gov/articles/PMC11230864/
2. Another review: https://arxiv.org/pdf/2209.15137
3. Review of biomarkers related to stress. https://onlinelibrary.wiley.com/doi/10.1002/dev.22490
4. ML study that predicts stress up to 1 min ahead: https://arxiv.org/abs/2106.07542
5. Relevant. See their Table 4 for a benchmark of WESAD data using different models. https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control
6. ML time series for stress duration prediction (predicts the duration of the next stress episode from a prior stress episode). Not directly related, but their method may be useful for us. https://dl.acm.org/doi/10.1145/2858036.2858218

### For stress vs exercise:
https://www.nature.com/articles/s41597-025-04845-9 and Dataset: https://physionet.org/content/wearable-device-dataset/1.0.0/

### For Personalization
https://arxiv.org/pdf/2107.05666 (Used WESAD)

------------------------------------------------

## Additional Datasets (explore after training the model):

**SWELL:** 

25 people performed typical knowledge work. A varied set of data was recorded: computer logging, facial expression from camera recordings, body postures from a Kinect 3D sensor and heart rate (variability) and skin conductance from body sensors. Dataset not only contains raw data, but also preprocessed data and extracted features. The participants' stress was assessed with validated questionnaires
https://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html 

**Exam Stress:**

data contains electrodermal activity, heart rate, blood volume pulse, skin surface temperature, inter beat interval and accelerometer data recorded during three exam sessions https://physionet.org/content/wearable-exam-stress/1.0.0/

**BIOSTRESS**
https://www.kaggle.com/datasets/orvile/biostress-dataset

