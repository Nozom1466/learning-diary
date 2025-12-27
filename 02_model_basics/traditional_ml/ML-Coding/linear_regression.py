import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _compute_gradients(self, X, y, y_pred):
        """Compute gradients: dL/dw = (2/m) * X^T * (y_pred - y), dL/db = (2/m) * sum(y_pred - y)"""
        m = X.shape[0]
        error = y_pred - y
        dw = (2 / m) * np.dot(X.T, error)
        db = (2 / m) * np.sum(error)
        return dw, db
    
    def _update_parameters(self, dw, db):
        """Update parameters: w = w - lr * dw, b = b - lr * db"""
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
    
    def _compute_loss(self, y_true, y_pred):
        """Mean Squared Error: MSE = (1/m) * sum((y_pred - y_true)^2)"""
        m = y_true.shape[0]
        return np.mean((y_pred - y_true) ** 2)
    
    def fit(self, X, y):
        """Train the linear regression model"""
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self._update_parameters(dw, db)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions: y = Xw + b"""
        return np.dot(X, self.weights) + self.bias


# Test Case 1: Simple linear relationship
print("=" * 50)
print("Test Case 1: Simple Linear Relationship")
print("=" * 50)

np.random.seed(42)
X_train = np.random.randn(100, 1)
y_train = 3 * X_train.squeeze() + 7 + np.random.randn(100) * 0.5

model1 = LinearRegression(learning_rate=0.1, n_iterations=1000)
model1.fit(X_train, y_train)

print(f"\nLearned parameters:")
print(f"Weight: {model1.weights[0]:.4f} (True: 3.0)")
print(f"Bias: {model1.bias:.4f} (True: 7.0)")

X_test = np.array([[1.0], [2.0], [3.0]])
predictions = model1.predict(X_test)
print(f"\nPredictions for X = [1, 2, 3]: {predictions}")


# Test Case 2: Multiple features
print("\n" + "=" * 50)
print("Test Case 2: Multiple Features")
print("=" * 50)

np.random.seed(123)
X_train_multi = np.random.randn(200, 3)
true_weights = np.array([2.5, -1.3, 4.0])
y_train_multi = np.dot(X_train_multi, true_weights) + 5 + np.random.randn(200) * 0.3

model2 = LinearRegression(learning_rate=0.1, n_iterations=1000)
model2.fit(X_train_multi, y_train_multi)

print(f"\nLearned weights: {model2.weights}")
print(f"True weights: {true_weights}")
print(f"Bias: {model2.bias:.4f} (True: 5.0)")


# Test Case 3: Perfect fit (no noise)
print("\n" + "=" * 50)
print("Test Case 3: Perfect Fit (No Noise)")
print("=" * 50)

X_perfect = np.array([[1], [2], [3], [4], [5]])
y_perfect = 2 * X_perfect.squeeze() + 1

model3 = LinearRegression(learning_rate=0.1, n_iterations=500)
model3.fit(X_perfect, y_perfect)

print(f"\nLearned parameters:")
print(f"Weight: {model3.weights[0]:.6f} (True: 2.0)")
print(f"Bias: {model3.bias:.6f} (True: 1.0)")
print(f"Final Loss: {model3.losses[-1]:.10f}")