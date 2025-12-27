import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _sigmoid(self, z):
        """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip for numerical stability
    
    def _compute_loss(self, y_true, y_pred):
        """Binary cross-entropy: L = -(1/m) * sum(y*log(ŷ) + (1-y)*log(1-ŷ))"""
        m = y_true.shape[0]
        epsilon = 1e-15  # prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _compute_gradients(self, X, y, y_pred):
        """Compute gradients: dL/dw = (1/m) * X^T * (ŷ - y), dL/db = (1/m) * sum(ŷ - y)"""
        m = X.shape[0]
        error = y_pred - y
        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)
        return dw, db
    
    def _update_parameters(self, dw, db):
        """Update parameters: w = w - lr * dw, b = b - lr * db"""
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
    
    def fit(self, X, y):
        """Train the logistic regression model"""
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
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
    
    def predict_proba(self, X):
        """Return probability predictions: P(y=1|X)"""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Return binary predictions: 0 or 1"""
        return (self.predict_proba(X) >= threshold).astype(int)


# Test Case 1: Linearly separable binary classification
print("=" * 50)
print("Test Case 1: Linearly Separable Data")
print("=" * 50)

np.random.seed(42)
# Class 0: centered at (-2, -2)
X_class0 = np.random.randn(50, 2) + np.array([-2, -2])
y_class0 = np.zeros(50)

# Class 1: centered at (2, 2)
X_class1 = np.random.randn(50, 2) + np.array([2, 2])
y_class1 = np.ones(50)

X_train = np.vstack([X_class0, X_class1])
y_train = np.hstack([y_class0, y_class1])

# Shuffle
indices = np.random.permutation(100)
X_train = X_train[indices]
y_train = y_train[indices]

model1 = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model1.fit(X_train, y_train)

predictions = model1.predict(X_train)
accuracy = np.mean(predictions == y_train)
print(f"\nTraining Accuracy: {accuracy:.4f}")

# Test on new samples
X_test = np.array([[-3, -3], [3, 3], [0, 0]])
probs = model1.predict_proba(X_test)
preds = model1.predict(X_test)
print(f"\nTest samples:")
for i, (x, prob, pred) in enumerate(zip(X_test, probs, preds)):
    print(f"  X={x}, P(y=1)={prob:.4f}, Prediction={pred}")


# Test Case 2: XOR-like pattern (non-linearly separable)
print("\n" + "=" * 50)
print("Test Case 2: Non-linearly Separable (XOR-like)")
print("=" * 50)

np.random.seed(123)
X_xor = np.random.randn(200, 2)
y_xor = ((X_xor[:, 0] > 0) ^ (X_xor[:, 1] > 0)).astype(int)  # XOR pattern

model2 = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model2.fit(X_xor, y_xor)

predictions2 = model2.predict(X_xor)
accuracy2 = np.mean(predictions2 == y_xor)
print(f"\nTraining Accuracy: {accuracy2:.4f}")
print("Note: Linear model struggles with XOR pattern (expected)")


# Test Case 3: Single feature classification
print("\n" + "=" * 50)
print("Test Case 3: Single Feature Classification")
print("=" * 50)

np.random.seed(456)
X_single = np.random.randn(100, 1)
y_single = (X_single.squeeze() > 0).astype(int)

model3 = LogisticRegression(learning_rate=0.5, n_iterations=500)
model3.fit(X_single, y_single)

predictions3 = model3.predict(X_single)
accuracy3 = np.mean(predictions3 == y_single)
print(f"\nTraining Accuracy: {accuracy3:.4f}")

# Decision boundary
print(f"\nDecision boundary (when z=0): x = {-model3.bias / model3.weights[0]:.4f}")
print(f"True boundary: x = 0")


# Test Case 4: Imbalanced classes
print("\n" + "=" * 50)
print("Test Case 4: Imbalanced Classes")
print("=" * 50)

np.random.seed(789)
X_imb_0 = np.random.randn(80, 2) - 1
X_imb_1 = np.random.randn(20, 2) + 1
X_imb = np.vstack([X_imb_0, X_imb_1])
y_imb = np.hstack([np.zeros(80), np.ones(20)])

indices_imb = np.random.permutation(100)
X_imb = X_imb[indices_imb]
y_imb = y_imb[indices_imb]

model4 = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model4.fit(X_imb, y_imb)

predictions4 = model4.predict(X_imb)
accuracy4 = np.mean(predictions4 == y_imb)
print(f"\nTraining Accuracy: {accuracy4:.4f}")
print(f"Class distribution: {np.sum(y_imb == 0)} class 0, {np.sum(y_imb == 1)} class 1")
print(f"Prediction distribution: {np.sum(predictions4 == 0)} class 0, {np.sum(predictions4 == 1)} class 1")