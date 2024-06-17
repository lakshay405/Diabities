import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading the diabetes dataset into a Pandas DataFrame
df_diabetes = pd.read_csv('data.xlsx')

# Displaying the first 5 rows of the dataset
print(df_diabetes.head())

# Number of rows and columns in the dataset
print(df_diabetes.shape)

# Getting statistical measures of the data
print(df_diabetes.describe())

# Counting the distribution of outcomes
print(df_diabetes['Outcome'].value_counts())

# Grouping by outcome to find mean values
print(df_diabetes.groupby('Outcome').mean())

# Separating the data into features (X) and target (Y)
X = df_diabetes.drop(columns='Outcome', axis=1)
Y = df_diabetes['Outcome']
print(X.head())
print(Y.head())

# Standardizing the data
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)

# Reassigning X with standardized data
X = X_scaled

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Initializing the Support Vector Machine classifier
classifier = svm.SVC(kernel='linear')

# Training the classifier on the training data
classifier.fit(X_train, Y_train)

# Accuracy score on the training data
Y_train_pred = classifier.predict(X_train)
training_accuracy = accuracy_score(Y_train_pred, Y_train)
print('Accuracy score on the training data: ', training_accuracy)

# Accuracy score on the test data
Y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(Y_test_pred, Y_test)
print('Accuracy score on the test data: ', test_accuracy)

# Example input data for prediction
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Converting input data to numpy array and reshaping it
input_data_array = np.asarray(input_data).reshape(1, -1)

# Standardizing the input data using the scaler
std_input_data = scaler.transform(input_data_array)
print(std_input_data)

# Making prediction using the trained classifier
prediction = classifier.predict(std_input_data)
print(prediction)

# Outputting prediction result based on class
if prediction[0] == 0:
    print('The person is not diabetic.')
else:
    print('The person is diabetic.')
