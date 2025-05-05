# Import the diabetes dataset from sklearn
from sklearn import datasets

# Load the diabetes dataset
data = datasets.load_diabetes()

# Import pandas for data manipulation
import pandas as pd

# Create a DataFrame from the dataset's features
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target variable (diabetes progression) to the DataFrame
df['target'] = pd.Series(data.target)

# Import numpy for numerical operations
import numpy as np

# Import seaborn and matplotlib for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Define the target variable (y) as the 'target' column
y = df['target']

# Define the features (X) by dropping the 'target' column
X = df.drop(['target'], axis=1)

# Import train_test_split to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Import GradientBoostingRegressor for regression tasks
from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting Regressor with hyperparameters
gbr = GradientBoostingRegressor(
    n_estimators=1000,  # Number of boosting stages
    learning_rate=0.01,  # Learning rate shrinks the contribution of each tree
    max_depth=3  # Maximum depth of each tree
)

# Train the Gradient Boosting Regressor on the training data
gbr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gbr.predict(X_test)

# Import mean_absolute_error to evaluate the model's performance
from sklearn.metrics import mean_absolute_error

# Calculate the Mean Absolute Error (MAE) between predicted and actual values
mae = mean_absolute_error(y_test, y_pred)

# Print the Mean Absolute Error to evaluate the model's accuracy
print('Mean Absolute Error:', mae)

# Calculate feature importance scores from the trained Gradient Boosting Regressor
feature_scores = pd.Series(gbr.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Print the feature importance scores to understand which features contribute most to the model
print(feature_scores)

# Visualize the feature importance scores using a bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)

# Add a label to the x-axis of the plot
plt.xlabel('Feature Importance')

# Display the plot
plt.show()