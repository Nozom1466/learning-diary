import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
    
    def _initialize_centroids(self, X):
        """Randomly initialize centroids from data points"""
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _compute_distances(self, X, centroids):
        """Compute Euclidean distances between points and centroids"""
        # Shape: (n_samples, n_clusters)
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        return distances
    
    def _assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid"""
        distances = self._compute_distances(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster by reinitializing
                centroids[k] = X[np.random.choice(X.shape[0])]
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """Compute sum of squared distances to nearest centroid"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """Fit K-means clustering"""
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        for i in range(self.max_iters):
            # Assignment step
            labels = self._assign_clusters(X, self.centroids)
            
            # Update step
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence: ||new_centroids - old_centroids|| < tol
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        # Store final labels and inertia
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels, self.centroids)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_


# Test Case 1: Simple 2D clustering
print("=" * 50)
print("Test Case 1: Simple 2D Clustering")
print("=" * 50)

np.random.seed(42)
# Generate 3 clusters
cluster1 = np.random.randn(30, 2) + np.array([0, 0])
cluster2 = np.random.randn(30, 2) + np.array([5, 5])
cluster3 = np.random.randn(30, 2) + np.array([0, 5])
X_train = np.vstack([cluster1, cluster2, cluster3])

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_train)

print(f"Converged centroids:\n{kmeans.centroids}")
print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")
print(f"Cluster assignments (first 10): {labels[:10]}")

# Test prediction on new points
X_test = np.array([[0, 0], [5, 5], [0, 5]])
predictions = kmeans.predict(X_test)
print(f"Predictions for test points {X_test.tolist()}: {predictions}")


# Test Case 2: 1D data
print("\n" + "=" * 50)
print("Test Case 2: 1D Clustering")
print("=" * 50)

X_1d = np.array([1, 2, 2.5, 8, 9, 10, 15, 16, 17]).reshape(-1, 1)
kmeans_1d = KMeans(n_clusters=3, random_state=42)
labels_1d = kmeans_1d.fit_predict(X_1d)

print(f"Data: {X_1d.flatten()}")
print(f"Centroids: {kmeans_1d.centroids.flatten()}")
print(f"Labels: {labels_1d}")


# Test Case 3: High-dimensional data
print("\n" + "=" * 50)
print("Test Case 3: High-Dimensional Clustering (5D)")
print("=" * 50)

np.random.seed(123)
X_high_dim = np.random.randn(100, 5)
X_high_dim[:40] += np.array([2, 2, 2, 2, 2])
X_high_dim[40:80] += np.array([-2, -2, -2, -2, -2])

kmeans_hd = KMeans(n_clusters=3, max_iters=100, random_state=42)
labels_hd = kmeans_hd.fit_predict(X_high_dim)

print(f"Data shape: {X_high_dim.shape}")
print(f"Centroids shape: {kmeans_hd.centroids.shape}")
print(f"Inertia: {kmeans_hd.inertia_:.2f}")
print(f"Cluster sizes: {np.bincount(labels_hd)}")
