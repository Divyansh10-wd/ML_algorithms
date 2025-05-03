# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs

# Load the datasets
test = pd.read_csv('test.csv')  # Load the test dataset
train = pd.read_csv('train.csv')  # Load the training dataset

# Combine the train and test datasets for preprocessing
df = pd.concat([train, test], sort=False)

# Check for missing values in the dataset
print(df.isnull().sum())

# Drop the 'Cabin' column as it has too many missing values
df = df.drop('Cabin', axis=1)

# Import SimpleImputer for handling missing values
from sklearn.impute import SimpleImputer

# Replace missing values in the 'Age' column with the mean
si = SimpleImputer(strategy='mean')
df['Age'] = si.fit_transform(df[['Age']])

# Replace missing values in the 'Survived' column with the mean
df['Survived'] = si.fit_transform(df[['Survived']])

# Replace missing values in the 'Embarked' column with the most frequent value
si1 = SimpleImputer(strategy='most_frequent')
df['Embarked'] = si1.fit_transform(df[['Embarked']]).ravel()

# Replace missing values in the 'Fare' column with the mean
df['Fare'] = si.fit_transform(df[['Fare']])

# Check for missing values again to ensure they are handled
print(df.isnull().sum())

# Drop unnecessary columns 'Name' and 'Ticket'
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Set a threshold for identifying outliers
threshold = 3

# Import zscore for outlier detection
from scipy.stats import zscore

# Specify numerical features for outlier detection
numerical_features = ['Age', 'Fare']

# Calculate z-scores for the numerical features
z_score = abs(zscore(df[numerical_features]))

# Import winsorize for handling outliers
from scipy.stats.mstats import winsorize

# Apply winsorization to the 'Age' column to limit extreme values
df['Age'] = winsorize(df['Age'], limits=[0.15, 0.15])

# Apply winsorization to the 'Fare' column to limit extreme values
df['Fare'] = winsorize(df['Fare'], limits=[0.15, 0.15])

# Identify outliers based on z-scores
outliers = np.where(z_score > 3)
print(outliers)

# Import OrdinalEncoder for encoding categorical variables
from sklearn.preprocessing import OrdinalEncoder

# Encode the 'Sex' column as ordinal values
oe = OrdinalEncoder()
df['Sex'] = oe.fit_transform(df[['Sex']])

# Encode the 'Embarked' column as ordinal values
df['Embarked'] = oe.fit_transform(df[['Embarked']])

# Encode the 'Survived' column as ordinal values
df['Survived'] = oe.fit_transform(df[['Survived']])

# Separate features (X) and target variable (y)
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

# Import train_test_split for splitting the dataset
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Import AdaBoostClassifier for training the model
from sklearn.ensemble import AdaBoostClassifier

# Initialize the AdaBoost classifier with 45 estimators and a learning rate of 1.0
ab = AdaBoostClassifier(n_estimators=45, learning_rate=1.0, random_state=0)

# Train the AdaBoost classifier on the training data
ab.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ab.predict(X_test)

# Import accuracy_score for evaluating the model
from sklearn.metrics import accuracy_score

# Calculate and print the accuracy of the AdaBoost classifier
print('Accuracy of AdaBoostClassifier: ', accuracy_score(y_test, y_pred) * 100)

