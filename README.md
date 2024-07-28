# customer-data
sensor_data.csv
timestamp	sensor1	sensor2	sensor3	sensor4	sensor5	failure
2023-01-01 00:00:00	10.5	7.8	5.6	3.2	2.1	0
2023-01-01 01:00:00	11.0	8.0	5.7	3.3	2.2	0
2023-01-01 02:00:00	10.8	7.9	5.8	3.4	2.3	0
2023-01-01 03:00:00	12.0	8.2	6.0	3.5	2.5	1
2023-01-01 04:00:00	10.5	7.8	5.6	3.2	2.1	0
2023-01-01 05:00:00	11.0	8.0	5.7	3.3	2.2	0
2023-01-01 06:00:00	10.8	7.9	5.8	3.4	2.3	0
2023-01-01 07:00:00	12.0	8.2	6.0	3.5	2.5	1
2023-01-01 08:00:00	10.5	7.8	5.6	3.2	2.1	0
2023-01-01 09:00:00	11.0	8.0	5.7	3.3	2.2	0
2023-01-01 10:00:00	10.8	7.9	5.8	3.4	2.3	0
2023-01-01 11:00:00	12.0	8.2	6.0	3.5	2.5	1
...	...	...	...	...	...	...

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Check for missing values
print(df.isnull().sum())

# Normalize the data
scaler = StandardScaler()
df[['Age', 'AnnualIncome', 'SpendingScore']] = scaler.fit_transform(df[['Age', 'AnnualIncome', 'SpendingScore']])


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the features
sns.pairplot(df[['Age', 'AnnualIncome', 'SpendingScore']])
plt.show()

from sklearn.cluster import KMeans

# Define the features for clustering
features = df[['AnnualIncome', 'SpendingScore']]

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Fit the KMeans model with the optimal number of clusters
optimal_clusters = 3  # Example, adjust based on the Elbow plot
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(features)

# Plot the clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()








