# Network Intrusion Detection with Decision Tree Classifier
This repository contains a project aimed at detecting various types of network intrusions using machine learning techniques.<br>
The project utilizes the NSL-KDD dataset to train and test a Decision Tree Classifier for identifying different types of network attacks.<br>


## Table of Contents  
### [Introduction](#introduction)  <br>
### [Dataset](#dataset)  <br>
### [Prerequisites](#prerequisites) <br>
### [Project Structure](#project_structure) <br>
### [Data Preprocessing](#data_preprocessing)<br>
### [Feature Selection](#feature_selection) <br>
### [Model Training and Evaluation](#model_training_and_evaluation) <br> 
### [Results](#results)<br>
### [Usage](#usage) <br>



## Introduction 
Network intrusion detection is a critical task for maintaining the security of computer networks.<br>
This project leverages the NSL-KDD dataset to train a Decision Tree Classifier that can detect various types of network attacks, 
such as DoS, Probe, R2L, and U2R attacks<br>

## Dataset
The project uses the NSL-KDD dataset, which is a refined version of the KDD'99 dataset.<br>
The dataset contains various features extracted from network traffic and labels indicating normal or specific attack types.<br>
Training Set: [NSL_KDD_Train.csv](https://github.com/Mth410/NSL-KDD/blob/main/NSL_KDD_Train.csv) <br>
Test Set: [NSL_KDD_Test.csv](https://github.com/Mth410/NSL-KDD/blob/main/NSL_KDD_Test.csv) <br>

## Prerequisites
The following Python libraries are required to run the code:<br>
numpy<br>
pandas<br>
matplotlib<br>
scikit-learn<br>

Install the required packages using pip: 
```
pip install numpy pandas matplotlib scikit-learn
```
## Project Structure
### The project is structured as follows:<br>

Data Loading: Load training and test datasets.<br>
Data Preprocessing: Assign feature names, check for missing values, and encode categorical features.<br>
Feature Selection: Use ANOVA F-test for univariate feature selection.<br>
Model Training: Train a Decision Tree Classifier for each attack type.<br>
Model Evaluation: Evaluate the classifier using confusion matrix and classification report.<br>

## Data Preprocessing
### 1. Loading Data: The NSL-KDD training and test datasets are loaded using pandas<br>
```python
   df_train = pd.read_csv("NSL_KDD_Train.csv")
   df_test = pd.read_csv("NSL_KDD_Test.csv")
```
### 2. Assigning Feature Names: Assign human-readable feature names to the dataset.
```python
   features = [...]
   train_dataset = pd.read_csv("NSL_KDD_Train.csv", header=None, names=features)
   test_dataset = pd.read_csv("NSL_KDD_Test.csv", header=None, names=features)
```
### 3. Check for Missing Values: Ensure there are no missing values in the datasets.
```python
   print(train_dataset.isnull().sum())
   print(test_dataset.isnull().sum())
```
### 4. Label Encoding: Encode categorical features using LabelEncoder and OneHotEncoder.
```python
   label_encoder = LabelEncoder()
   onehot_encoder = OneHotEncoder()
```
### 5. Feature Engineering: Transform categorical features into numerical values and handle discrepancies between training and test sets.
```python
   train_dataset_categorical_values_enc = train_dataset_categorical_values.apply(LabelEncoder().fit_transform)
   train_dataset_categorical_values_encoded = onehot_encoder.fit_transform(train_dataset_categorical_values_enc)
```
## Feature Selection
### Use ANOVA F-test for univariate feature selection to select the most informative features.
```python
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=20)
X_newDoS = selector.fit_transform(X_DoS, Y_DoS)
```
## Model Training and Evaluation
### 1. Train-Test Split: Split the dataset into training and testing sets
```python
X_trainDoS, X_testDoS, y_trainDoS, y_testDoS = train_test_split(X_newDoS, Y_DoS, test_size=0.2, random_state=42)
```
### 2. Train Classifier: Train a Decision Tree Classifier for each attack type.
```python
clf_DoS = DecisionTreeClassifier(random_state=42)
clf_DoS.fit(X_trainDoS, y_trainDoS)
```
### 3. Model Evaluation: Evaluate the model using classification report and confusion matrix.
```python
Y_DoS_pred = clf_DoS.predict(X_testDoS)
print(classification_report(y_testDoS, Y_DoS_pred))
```
## Results
### The results include:

* #### Distribution of detected threats.<br>
* #### Classification reports for each type of attack.<br>
* #### Evaluation metrics such as accuracy, precision, recall, and F1-score.<br>
* #### Decision trees visualizations for each type of attack.

## Usage
To run the project, follow these steps:<br>
1- Clone the repository:
```
git clone https://github.com/OmarAbdelall/NSL-KDD.git 
cd NSL-KDD
```
2- Ensure you have the required datasets (NSL_KDD_Train.csv and NSL_KDD_Test.csv) in the project directory.<br>
3- Execute the provided code in a Python environment. You can use Jupyter Notebook, Google Colab, or any Python IDE.<br>
4- After running the code, review the results presented in the output. This includes data preprocessing, feature selection, model training, and evaluation metrics.
