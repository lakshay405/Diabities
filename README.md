# Diabities
Diabetes Prediction using Support Vector Machine (SVM)
This project focuses on predicting the onset of diabetes in individuals using machine learning techniques, specifically Support Vector Machine (SVM). The goal is to build a classifier that can effectively distinguish between diabetic and non-diabetic individuals based on various health metrics.

Dataset
The dataset (data.xlsx) contains health information of individuals, including attributes such as glucose level, blood pressure, BMI, and age, along with a binary target variable indicating the presence (1) or absence (0) of diabetes.

Workflow
Data Loading and Preprocessing:

Load the diabetes dataset from an Excel file into a Pandas DataFrame (df_diabetes).
Display the first few rows of the dataset, check dimensions, and gather statistical summaries to understand the data distribution.
Exploratory Data Analysis (EDA):

Explore the distribution of outcomes (diabetic vs. non-diabetic) using value counts and mean values grouped by outcome.
Feature Standardization:

Standardize the features using StandardScaler to ensure all features have a mean of 0 and a standard deviation of 1, which is crucial for SVM models.
Model Training and Evaluation:

Separate features (X) and target (Y) from the dataset.
Split the data into training and testing sets using train_test_split.
Initialize and train a Support Vector Machine classifier (svm.SVC) with a linear kernel.
Evaluate the model's performance using accuracy scores on both training and testing sets.
Prediction:

Demonstrate prediction capabilities by inputting example data and predicting the likelihood of diabetes using the trained SVM model.
Output the prediction result based on the predicted class.
Libraries Used
numpy and pandas for data manipulation and analysis.
sklearn for preprocessing (StandardScaler), model selection (train_test_split, svm.SVC), and evaluation (accuracy_score).
Conclusion
This project showcases the application of SVM for predicting diabetes based on health metrics. By utilizing a well-structured dataset and standardizing features, the SVM classifier achieves accurate predictions, providing valuable insights into diabetic risk assessment and potentially aiding in early intervention and healthcare management.
