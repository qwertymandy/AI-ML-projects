import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Load MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# For faster computation, use a subset (e.g., 10000 samples)
X_sample = X[:10000]
y_sample = y[:10000].astype(int)

# Optional: reduce dimensionality for visualization and speed
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_sample)

# Initialize KMeans with 10 clusters (digits 0-9)
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_pca)

# Cluster labels
clusters = kmeans.labels_

# Map clusters to true digit labels to evaluate clustering
def cluster_accuracy(y_true, y_pred):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Assign most common label in each cluster as that cluster's label
    import scipy.optimize
    indexes = scipy.optimize.linear_sum_assignment(-cm)
    mapping = {old: new for old, new in zip(indexes[1], indexes[0])}
    y_pred_mapped = np.array([mapping.get(label) for label in y_pred])
    accuracy = np.mean(y_pred_mapped == y_true)
    return accuracy, y_pred_mapped

accuracy, y_pred_mapped = cluster_accuracy(y_sample, clusters)
print(f"KMeans clustering accuracy (after best label mapping): {accuracy:.4f}")

# Plot some cluster centers as images
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
centers = kmeans.cluster_centers_.reshape(10, 5, 10)  # Reshape PCA centers approx

for i, ax in enumerate(axes.flat):
    ax.imshow(centers[i], cmap='viridis')
    ax.set_title(f"Cluster {i}")
    ax.axis('off')

plt.suptitle("Cluster Centers Visualized in PCA Space")
plt.show()
