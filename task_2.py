import pandas as pd
from utils import * 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt 

merged_df = pd.read_csv('data.csv')
merged_df.drop(index = 0, inplace = True)

convert_columns_to_numeric(merged_df)
new_df = replace_outliers_with_mad(merged_df, special_cols= {'SCR cr': 'mean'})
data = new_df.drop(columns='Firm').values

# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Finding the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(2, 11)  # Testing for k values from 2 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
    
# Plotting the Elbow Method results
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# k = 3 shows the most slope, albeit small. 

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)

# Getting the centroids
centroids = kmeans.cluster_centers_

# Inverse transform the centroids to get them back to the original scale
centroids_original_scale = scaler.inverse_transform(centroids)

new_df['Cluster'] = kmeans.labels_

# Grabbing code from task 1 to get average aggregated stats 

new_df['Avg SCR cr'] = new_df.filter(regex='SCR cr').mean(axis = 1)
new_df['Avg NCR'] = new_df.filter(regex='NCR').mean(axis=1)

years = ['2016', '2017', '2018', '2019', '2020']

small_value = 1e-5
for year in years:
    gwp_col = f'GWP {year}'
    nwp_col = f'NWP {year}'
    ratio_col = f'GWP/NWP Ratio {year}'

    # Adding small value to NWP and ensuring NWP is not extremely small
    new_df[nwp_col] = new_df[nwp_col].apply(lambda x: x + small_value if x < small_value else x)

    # Calculating the ratio
    new_df[ratio_col] = new_df[gwp_col] / new_df[nwp_col]

# Replace 'inf' values with NaN
new_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Calculating the average GWP/NWP ratio across years
new_df['Avg GWP/NWP'] = new_df.filter(regex='GWP/NWP Ratio').mean(axis=1)

# Which cluster do high NCR firms fall in 

# from task 1 we know this
high_ncr_firms = ['Firm 210', 'Firm 305', 'Firm 157', 'Firm 40', 'Firm 299']
new_df[new_df['Firm'].isin(high_ncr_firms)]['Cluster']

low_scr_firms = ['Firm 109', 'Firm 141', 'Firm 183', 'Firm 225', 'Firm 251']
new_df[new_df['Firm'].isin(low_scr_firms)]['Cluster']

new_df.groupby('Cluster', as_index=False)[['NWP 2020', 'GWP 2020', 'EOAOL 2020', 'NCR 2020','SCR coverage ratio 2020']].aggregate(['mean'])