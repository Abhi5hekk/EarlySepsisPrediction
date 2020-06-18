# Early Prediction of Sepsis from Clinical Data
Probabilistic Time Series dependent on the PhysioNet Computing in Cardiology Challenge 2019. [Link](https://physionet.org/content/challenge-2019/1.0.0/)

## Introduction
Sepsis is a life-threatening illness caused by your body’s response to an infection. Your immune system protects you from many illnesses and infections, but it’s also possible for it to go into overdrive in response to an infection. Sepsis develops when the chemicals the immune system releases into the bloodstream to fight an infection cause inflammation throughout the entire body instead. Severe cases of sepsis can lead to septic shock, which is a medical emergency. Around 30 millions individuals create sepsis and one-fifth of them die from the infection consistently around the globe. Distinguishing sepsis early and beginning quick treatment frequently spare patients lives.

## Project Objective
The objective of this project is to early detect sepsis (6 hours ahead) utilizing physiological information. The sources of info are patients' data, including imperative signs, research center qualities and demographics. The output is the results whether the model predicts non-Sepsis patients or Sepsis patients six hours in front of clinical beginning time.

## Dataset
For this study, we use clinical data of ICU patients from two separate hospital systems provided by the PhysioNet Computing in Cardiology Challenge 2019. The data for each patient are saved in a single pipe-delimited text file that has a fixed header. Each row of a patient file represents a single hour's worth for all the measurements within that ICU-hour stay. These measurements include vital signs, laboratory, and demographics values of 40 time-dependent variables. Nan indicates that the measurement is missing at this time interval. In total, we used over 40,336 patient files. After concatenating ICU-hour-stay entries from all the patients, we have about 1,552,229 lines of data in total.

According to the Challenge, labels in the dataset already take the goal of predicting Sepsis. The label for each hour of patient data is 1 (Sepsis onset positive) or 0 (Sepsis onset negative). Summarized from the labels, we have a very imbalanced dataset.

[Click here](https://archive.physionet.org/users/shared/challenge-2019/) to download the complete training database, consisting of two parts: training set A (20,336 subjects) and B (20,000 subjects).
## Data Preprocessing
#### Train-Validation-Test Split
First approach for solving early prediction of sepsis is to ignore the temporal component in the data. So each record can be treated as a single entity not dependent on any other record. This assumption might be flawed but for establishing a baseline, we are taking this assumption.<br/>

Train : 30K Patients Test : 5K Patients Validation : 5K Patients
#### Distribution of Dependent Variable
In the training dataset, we have a very imbalanced dataset. In label 0 we have about 1147309 lines of data and label 1 we have about 22556 lines of data.<br/>
![Screenshot](img.JPG)<br/>
As we can see from the plot, it is a case of severe class imbalance . There are multiple methods that we could try to balance it (Over sampling or Undersampling) or even proceed without balancing but change the evaluation metric to average precision or roc_auc_sore.
#### Check For null in Features/ Independent variables
![Screenshot](img1.JPG)<br/>
We see that most of the columns have 90% + missing data. There are multiple ways of dealing with it. To establish a concrete baseline, I have decided to remove features with 90 % or more missing data.<br/>
Remaining features: 'HR',
 'O2Sat',
 'Temp',
 'SBP',
 'MAP',
 'DBP',
 'Resp',
 'FiO2',
 'Glucose',
 'Age',
 'HospAdmTime',
 'ICULOS',
 'Gender',
 'Unit1',
 'Unit2'
## Baseline
The Features which are used for basleline. No Feature engineering has been done yet. We create the preprocessing pipelines for both numeric and categorical data.
```python
cont_scale_pipeline = make_pipeline(SimpleImputer(strategy = "median"), 
                                    StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy = "constant", 
                                           fill_value = 999), 
                             OneHotEncoder(handle_unknown="ignore"))
preprocess_trans_scale = make_column_transformer((cont_scale_pipeline, 
                                                  ~categorical), 
                                                 (cat_pipeline, categorical))
```
You can see that we have both categorical and numeric variables so as a minimum we have applied a one hot encoding transformation and some sort of scaler. we have used a scikit-learn pipeline to perform those transformations. The first step in building the pipeline is to define each transformer type. The convention here is generally to create transformers for the different variable types. In the code above we have created a numeric transformer which applies a StandardScaler, and includes a SimpleImputer to fill in any missing values. This is a really nice function in scikit-learn and has a number of options for filling missing values. I have chosen to use median but another method may result in better performance. The categorical transformer also has a SimpleImputer with a different fill method, and leverages OneHotEncoder to transform the categorical values into integers. Next we use the ColumnTransformer to apply the transformations to the correct columns in the dataframe.<br/>

Finally, the preprocessing pipeline is integrated in a full prediction pipeline, together with a simple classification models:
* LogisticRegression
* DecisionTree
* GradientBoosting
* RandomForest
* Gaussian mixture model
* Multilayer Perceptron

#### Results
![Screenshot](img3.JPG)
## Feature Engineering
#### Heart Rate
Heart rate for a healthy adult is between 60 and 100. For a healthy infant it is between 70 and 190. Creating a new feature custom_hr , which is categorical variable having three values Normal, Abnormal and Missing.
```python

def feature_engineer_hr(train):
    train.loc[(train['HR'] >= 100) & (train['Age'] >= 10 ),
            'custom_hr'] = 'abnormal'
    train.loc[(train['HR'] < 100) & (train['HR'] > 60) & (train['Age'] >= 10 ),
            'custom_hr'] = 'normal'
    train.loc[(train['HR'] >= 70) & (train['HR'] < 190) & (train['Age'] < 10 ),
            'custom_hr'] = 'normal'
    train.loc[((train['HR'] < 70) | (train['HR'] >= 190)) & (train['Age'] < 10 ),
            'custom_hr'] = 'abnormal'
    train['custom_hr'].fillna('Missing', inplace=True)
    return train
```
#### Temperature
Temperature for a healthy human being is between 36.4 degree C to 37.6 degree C. Creating a new feature custom_temp , which is categorical variable having three values Normal, Abnormal and Missing.
```python 
def feature_engineer_temp(train):
    train.loc[(train['Temp'] >= 36.4) & (train['Temp'] < 37.6), 
            'custom_temp'] = 'normal'
    train.loc[(train['Temp'] < 36.4) | (train['Temp'] >= 37.6), 
            'custom_temp'] = 'abnormal'

    train['custom_temp'].fillna('Missing', inplace=True)
    return train
 ```
#### Age
Categorizing patient based on age to old, infant and Child/adult.
```python
def featuer_engineer_age(train):
    train.loc[train['Age'] >=65, 'custom_age'] = 'old'
    train.loc[train['Age'] <1, 'custom_age'] = 'infant'
    train.loc[(train['Age'] >=1) & (train['Age'] <65), 
            'custom_age'] = 'child/adult'
    return train
```
#### O2Stat
O2Stat for a healthy adult is between 90 and 100 for healthy human beings. Create a new categorical variable custom_o2stat with levels normal, abnormal and missing.
```python
def feature_engineer_o2stat(train):
    train.loc[(train['O2Sat'] >= 90) & (train['O2Sat'] < 100), 
            'custom_o2stat'] = 'normal'
    train.loc[(train['O2Sat'] < 90) & (train['O2Sat'] >= 0), 
            'custom_o2stat'] = 'abnormal'

    train['custom_o2stat'].fillna('Missing', inplace=True)
    return train
```
#### SBP and DBP
SBP stands for Systolic blood pressure, It is the upper number while measuring Blood pressure. DBP stands for Diastolic blood pressure , It is the lower number while measuring Blood pressure. Using both these columns to categorize blood pressure as low, normal, elevated , high and missing.
```python
def feature_engineer_blood_pressure(train):
    train.loc[(train['SBP'] <90) & (train['DBP'] <60), 'custom_bp'] = 'low'

    train.loc[(train['SBP'].between(90,120, inclusive=True)) & 
            (train['DBP'].between(60,80, inclusive=True)), 
            'custom_bp'] = 'normal'


    train.loc[(train['SBP'].between(120,140, inclusive=True)) & 
            (train['DBP'].between(80,90, inclusive=True)), 
            'custom_bp'] = 'elevated'


    train.loc[(train['SBP'] > 140 ) & 
            (train['DBP'] > 90 ), 'custom_bp'] = 'high'

    train['custom_bp'].fillna('Missing', inplace=True)
    return train
 ```
#### Respiration Rate
Respiration rate for healthy adults is between 12 and 20. Categorizing respiratory rate as normal and abnormal based on thresholds.
```python
def feature_engineer_resp_rate(train):
    train.loc[(train['Resp'].between(30,60)) & (train['Age'] <1), 
            'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 30) | (train['Resp'] > 60)) & 
             (train['Age'] <1) ,'custom_resp'] = 'abnormal'


    train.loc[(train['Resp'].between(24,40)) & (train['Age'].between(1,3)), 
            'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 24) | (train['Resp'] > 40)) & 
             (train['Age'].between(1,3)) ,'custom_resp'] = 'abnormal'


    train.loc[(train['Resp'].between(22,34)) & (train['Age'].between(3,6)), 
            'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 22) | (train['Resp'] > 34)) & 
             (train['Age'].between(3,6)) ,'custom_resp'] = 'abnormal'


    train.loc[(train['Resp'].between(18,30)) & (train['Age'].between(6,12)), 
            'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 18) | (train['Resp'] > 30)) & 
             (train['Age'].between(6,12)) ,'custom_resp'] = 'abnormal'


    train.loc[(train['Resp'].between(12,20)) & (train['Age'] >12), 
            'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 12) | (train['Resp'] > 20)) & (train['Age'] >12),
            'custom_resp'] = 'abnormal'

    train['custom_resp'].fillna('Missing', inplace=True)


    return train
```
#### Feature Selection
According to the CDC website heart rate, fever and BP are the most important signs of sepsis. So selecting these features with the same intution
```python
filtered_columns = ['Gender', 'custom_hr', 'custom_temp','custom_age', 
                    'custom_o2stat', 'custom_bp','custom_resp' ,'ICULOS', 
                    'HospAdmTime']
```

## Final Results
Finally, after fearture engineering again the preprocessing pipeline is integrated in a full prediction pipeline, together with a simple classification models:
* Logistic Regression
* Decision Tree
* Random Forest<br/>
And Grid-search is used to find the optimal hyperparameters of a models which results in the most 'accurate' predictions.

|  Model   | Recall | Precision | Avg. precision |
|----------|--------|-----------|----------------|
|Logistic Regression|0.005603|0.075377|0.014346|
|Decision Tree|0.084796|0.048901|0.016961|
|Randon Forest|0.027269|0.059543|0.015244|

## Future Work
In Future, we will going to proposed Deep Learning Models.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
## Author
* [ABHISHEK KUMAR](https://github.com/Abhi5hekk/) 16074002
* [RISHIKA RATHORE](https://github.com/Rishika-Rathore/) 17045084
* BHANU PRAKASH 16074004
