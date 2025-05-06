# Import necessary libraries
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced visualizations
import datetime as dt  # For date and time operations
import numpy as np  # For numerical operations
import warnings  # To suppress warnings
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output

# Import machine learning libraries
import sklearn
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.cluster import KMeans  # For K-Means clustering

# Load the dataset
df = pd.read_csv('OnlineRetail.csv', encoding='unicode_escape')  # Load the retail dataset
df['CustomerID'] = df['CustomerID'].astype(str)  # Convert CustomerID to string for consistency

# Calculate the total amount for each transaction
df['Amount'] = df['Quantity'] * df['UnitPrice']

# Group by CustomerID to calculate the total amount spent by each customer
df_m = df.groupby('CustomerID')['Amount'].sum()
df_m = df_m.reset_index()  # Reset index to convert the grouped data into a DataFrame

# Group by CustomerID to calculate the frequency of transactions for each customer
df_f = df.groupby('CustomerID')['InvoiceDate'].count()
df_f = df_f.reset_index()  # Reset index to convert the grouped data into a DataFrame
df_f.columns = ['CustomerID', 'Frequency']  # Rename columns for clarity

# Merge the frequency and amount data into a single DataFrame
df_retail = pd.merge(df_f, df_m, on='CustomerID', how='inner')

# Convert InvoiceDate to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')

# Calculate the recency (days since the last transaction) for each customer
df['Diff'] = df['InvoiceDate'].max() - df['InvoiceDate']
df_p = df.groupby('CustomerID')['Diff'].min()
df_p = df_p.reset_index()  # Reset index to convert the grouped data into a DataFrame
df_p['Diff'] = df_p['Diff'].dt.days  # Convert timedelta to days

# Merge the recency data into the retail DataFrame
df_retail = pd.merge(df_retail, df_p, on='CustomerID', how='inner')
df_retail.columns = ['CustomerID', 'Frequency', 'Amount', 'Recency']  # Rename columns for clarity

# Define the attributes for analysis
attributes = ['Amount', 'Frequency', 'Recency']

# OPTIONAL: Visualize the distribution of outliers using a boxplot
plt.rcParams['figure.figsize'] = [10, 8]
sns.boxplot(data=df_retail[attributes], orient="v", palette="Set2", whis=1.5, saturation=1.5, width=0.7)
plt.title("Outliers Variable Distribution", fontsize=14, fontweight='bold')
plt.ylabel("Range", fontweight='bold')
plt.xlabel("Attributes", fontweight='bold')
plt.show()

# REMOVE OUTLIERS FROM AMOUNT, FREQUENCY, AND RECENCY BY CALCULATING 5TH AND 95TH PERCENTILE

# Remove outliers from the 'Amount' column
Q1 = df_retail['Amount'].quantile(0.05)
Q3 = df_retail['Amount'].quantile(0.95)
IQR = Q3 - Q1
df_retail = df_retail[~((df_retail['Amount'] < (Q1 - 1.5 * IQR)) & (df_retail['Amount'] > (Q3 + 1.5 * IQR)))]

# Remove outliers from the 'Frequency' column
Q1 = df_retail['Frequency'].quantile(0.05)
Q3 = df_retail['Frequency'].quantile(0.95)
IQR = Q3 - Q1
df_retail = df_retail[~((df_retail['Frequency'] < (Q1 - 1.5 * IQR)) & (df_retail['Frequency'] > (Q3 + 1.5 * IQR)))]

# Remove outliers from the 'Recency' column
Q1 = df_retail['Recency'].quantile(0.05)
Q3 = df_retail['Recency'].quantile(0.95)
IQR = Q3 - Q1
df_retail = df_retail[~((df_retail['Recency'] < (Q1 - 1.5 * IQR)) & (df_retail['Recency'] > (Q3 + 1.5 * IQR)))]

# Select the features for clustering
df1 = df_retail[['Frequency', 'Amount', 'Recency']]

# Scale the features using StandardScaler
scaler = StandardScaler()
df_retail_scaled = scaler.fit_transform(df1)
df_retail_scaled = pd.DataFrame(df_retail_scaled, columns=['Frequency', 'Amount', 'Recency'])

# OPTIONAL: Determine the optimal number of clusters using the elbow method
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i, max_iter=50)
    kmeans.fit(df_retail_scaled)
    ssd.append(kmeans.inertia_)
plt.plot(range_n_clusters, ssd)
plt.show()

# FINAL MODEL: Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(df_retail_scaled)
df_retail['Cluster_id'] = kmeans.labels_  # Assign cluster labels to the original DataFrame

# Prepare data for visualization
X = df_retail.iloc[:, [1, 3]].values  # Select 'Frequency' and 'Recency' for visualization
y_means = kmeans.fit_predict(X)  # Predict cluster labels

# Visualize the clusters
plt.figure(figsize=(16, 7))
sns.scatterplot(x=X[y_means == 0, 0], y=X[y_means == 0, 1], color='red', s=50, label='Cluster 1')
sns.scatterplot(x=X[y_means == 1, 0], y=X[y_means == 1, 1], color='blue', s=50, label='Cluster 2')
sns.scatterplot(x=X[y_means == 2, 0], y=X[y_means == 2, 1], color='green', s=50, label='Cluster 3')
plt.xlabel("Amount")
plt.ylabel("Recency")
sns.scatterplot(x=kmeans.cluster_centers_[:4, 0], y=kmeans.cluster_centers_[:4, 1], color='black', s=200, label='Centroids')
plt.show()