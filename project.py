import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('3.txt', sep='\t', header=None)

# Assign column names based on the dataset description
column_names = [
    'video_id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments'
] + [f'related_id_{i}' for i in range(1, 21)]  # Assign Column Values along with related ID's

data.columns = column_names  # Assign column names to the DataFrame

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Select numerical features for clustering and correlation analysis
features = ['age', 'length', 'views', 'rate', 'ratings', 'comments']
X = data[features]

# Fill missing values with 0
X = X.fillna(0)

# Remove outliers using the IQR method (if remove_outliers_flag is True)
remove_outliers_flag = True  # Set to True to remove outliers, False to keep them

def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

if remove_outliers_flag:
    data = data.copy()
    for feature in features:
        data = remove_outliers(data, feature)  # Remove outliers if the flag is True
else:
    data = data  # Keep the original data if the flag is False

# Drop rows with missing values in the selected features
data = data.dropna(subset=features)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Perform PCA without reducing dimensions (keep all components)
pca_full = PCA()  # No n_components specified, so it keeps all components
X_pca_full = pca_full.fit_transform(X_scaled)  # Fit and transform the scaled data

# Calculate the explained variance ratio for each component
explained_variance_ratio = pca_full.explained_variance_ratio_

# Calculate the cumulative explained variance ratio
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Plot the explained variance ratio and cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', label='Cumulative Explained Variance')
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, label='Individual Explained Variance')

# Add a horizontal line at 95% variance
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained Variance')

# Annotate the point where cumulative variance reaches 95%
n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} Components for 95% Variance')

# Add labels and title
plt.title('Explained Variance Ratio by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.grid()
plt.show()

# Print the number of components needed to capture 95% of the variance
print(f"Number of components to capture 95% of the variance: {n_components_95}")

# Perform PCA with the optimal number of components
pca = PCA(n_components=n_components_95)  # Use the optimal number of components
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to the DataFrame
for i in range(n_components_95):
    data[f'pca_{i+1}'] = X_pca[:, i]

# Display the first few rows of the dataset with PCA components
print("\nDataFrame with PCA components:")
print(data.head())

# Clustering using K-Means
# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k (Outliers Removed)' if remove_outliers_flag else 'Elbow Method for Optimal k (With Outliers)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Visualize clustering for different k values
k_values = [3, 4, 5, 6]  # Different k values to visualize
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Clustering with Different K Values (PCA Visualization, Outliers Removed)' if remove_outliers_flag else 'Clustering with Different K Values (PCA Visualization, With Outliers)', fontsize=16)

for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42)  # Fit KMeans algorithm
    data[f'cluster_k{k}'] = kmeans.fit_predict(X_scaled)  # Predict clusters
    
    ax = axes[i//2, i%2]  # Create subplots
    sns.scatterplot(x='pca_1', y='pca_2', hue=f'cluster_k{k}', data=data, palette='viridis', s=100, ax=ax)  # Scatter plot
    ax.set_title(f'K = {k}')  # Set title
    ax.set_xlabel('PC1')  # Set x-axis label
    ax.set_ylabel('PC2')  # Set y-axis label
    ax.legend(title='Cluster')  # Add legend

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
plt.show()  # Display the plot

# Extract PCA loadings (weights of each feature in the principal components)
loadings = pca.components_  # Get the loadings

# Create a DataFrame for the loadings
loadings_df = pd.DataFrame(loadings, columns=features, index=[f'PC{i+1}' for i in range(n_components_95)])

# Display the loadings
print("\nPCA Loadings:")
print(loadings_df)

# Visualize the loadings using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('PCA Loadings (Feature Contributions to Principal Components)')
plt.show()

# Correlation Analysis
# Compute the correlation matrix with additional features
corr_matrix = data[features].corr()

# Plot the expanded correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Expanded Correlation Heatmap (Outliers Removed)' if remove_outliers_flag else 'Expanded Correlation Heatmap (With Outliers)')
plt.show()

# Analyze cluster characteristics for the optimal k
optimal_k = 3  # Choose the optimal k based on the Elbow Method
cluster_summary = data.groupby(f'cluster_k{optimal_k}')[features].mean()
print("\nCluster Summary for Optimal K (Outliers Removed):" if remove_outliers_flag else "\nCluster Summary for Optimal K (With Outliers):")
print(cluster_summary)