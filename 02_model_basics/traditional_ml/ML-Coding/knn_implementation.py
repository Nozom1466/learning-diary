import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        K-Nearest Neighbors Classifier
        
        Args:
            k: number of neighbors to consider
            distance_metric: 'euclidean' or 'manhattan'
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _compute_distance(self, x1, x2):
        """Compute distance between two points"""
        if self.distance_metric == 'euclidean':
            # L2 norm: sqrt(sum((x1 - x2)^2))
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            # L1 norm: sum(|x1 - x2|)
            return np.sum(np.abs(x1 - x2))
    
    def _predict_single(self, x):
        """Predict class for a single sample"""
        # Compute distances to all training samples
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Return most common label (majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """Predict classes for multiple samples"""
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ===== Test Cases =====

def test_simple_2d():
    """Test with simple 2D data"""
    print("=" * 50)
    print("Test 1: Simple 2D Classification")
    print("=" * 50)
    
    # Training data: two classes
    X_train = np.array([
        [1, 1], [1, 2], [2, 1],  # Class 0
        [5, 5], [5, 6], [6, 5]   # Class 1
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Test data
    X_test = np.array([
        [1.5, 1.5],  # Should be class 0
        [5.5, 5.5]   # Should be class 1
    ])
    y_test = np.array([0, 1])
    
    # Train and predict
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data: {X_test}")
    print(f"Predictions: {predictions}")
    print(f"True labels: {y_test}")
    print(f"Accuracy: {knn.score(X_test, y_test) * 100:.2f}%")
    print()


def test_iris_like():
    """Test with Iris-like dataset"""
    print("=" * 50)
    print("Test 2: Multi-class Classification (Iris-like)")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Generate 3 classes with different centers
    n_samples_per_class = 20
    
    # Class 0: centered at (1, 1)
    class_0 = np.random.randn(n_samples_per_class, 2) * 0.5 + [1, 1]
    
    # Class 1: centered at (5, 5)
    class_1 = np.random.randn(n_samples_per_class, 2) * 0.5 + [5, 5]
    
    # Class 2: centered at (1, 5)
    class_2 = np.random.randn(n_samples_per_class, 2) * 0.5 + [1, 5]
    
    # Combine data
    X = np.vstack([class_0, class_1, class_2])
    y = np.hstack([
        np.zeros(n_samples_per_class),
        np.ones(n_samples_per_class),
        np.ones(n_samples_per_class) * 2
    ])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split train/test
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Test different k values
    for k in [1, 3, 5]:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        print(f"k={k}: Accuracy = {accuracy * 100:.2f}%")
    
    print()


def test_distance_metrics():
    """Test different distance metrics"""
    print("=" * 50)
    print("Test 3: Comparing Distance Metrics")
    print("=" * 50)
    
    # Simple dataset
    X_train = np.array([
        [0, 0], [1, 1], [2, 2],  # Class 0
        [5, 5], [6, 6], [7, 7]   # Class 1
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    X_test = np.array([[1.5, 1.5], [6, 6]])
    y_test = np.array([0, 1])
    
    for metric in ['euclidean', 'manhattan']:
        knn = KNN(k=3, distance_metric=metric)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = knn.score(X_test, y_test)
        print(f"{metric.capitalize()} distance:")
        print(f"  Predictions: {predictions}")
        print(f"  Accuracy: {accuracy * 100:.2f}%")
    
    print()


def test_edge_cases():
    """Test edge cases"""
    print("=" * 50)
    print("Test 4: Edge Cases")
    print("=" * 50)
    
    # Test with k=1
    X_train = np.array([[1, 1], [2, 2], [3, 3]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[1.1, 1.1]])
    
    knn = KNN(k=1)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(f"k=1, Test point close to [1,1]: Prediction = {pred[0]}, Expected = 0")
    
    # Test with higher dimensional data
    X_train_5d = np.random.randn(30, 5)
    y_train_5d = np.random.randint(0, 3, 30)
    X_test_5d = np.random.randn(5, 5)
    
    knn_5d = KNN(k=5)
    knn_5d.fit(X_train_5d, y_train_5d)
    predictions_5d = knn_5d.predict(X_test_5d)
    print(f"\n5D data test:")
    print(f"  Training shape: {X_train_5d.shape}")
    print(f"  Test shape: {X_test_5d.shape}")
    print(f"  Predictions: {predictions_5d}")
    print()


if __name__ == "__main__":
    test_simple_2d()
    test_iris_like()
    test_distance_metrics()
    test_edge_cases()
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)
