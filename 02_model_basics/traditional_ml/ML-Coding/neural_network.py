import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, n_iterations=1000):
        """
        layer_sizes: list of integers [input_size, hidden1, hidden2, ..., output_size]
        """
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.n_layers = len(layer_sizes)
        self.weights = {}
        self.biases = {}
        self.losses = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights with Xavier initialization and biases with zeros"""
        for i in range(1, self.n_layers):
            self.weights[i] = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * np.sqrt(2.0 / self.layer_sizes[i-1])
            self.biases[i] = np.zeros((1, self.layer_sizes[i]))
    
    def _sigmoid(self, z):
        """Sigmoid: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _sigmoid_derivative(self, a):
        """Derivative: σ'(z) = σ(z) * (1 - σ(z))"""
        return a * (1 - a)
    
    def _relu(self, z):
        """ReLU: max(0, z)"""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """Derivative: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    def _forward_propagation(self, X):
        """Forward pass through the network"""
        cache = {'A0': X}
        A = X
        
        for i in range(1, self.n_layers):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            
            # Use ReLU for hidden layers, sigmoid for output layer
            if i < self.n_layers - 1:
                A = self._relu(Z)
            else:
                A = self._sigmoid(Z)
            
            cache[f'Z{i}'] = Z
            cache[f'A{i}'] = A
        
        return A, cache
    
    def _compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _backward_propagation(self, y, cache):
        """Backward pass: compute gradients using chain rule"""
        m = y.shape[0]
        gradients = {}
        
        # Output layer gradient: dL/dZ = ŷ - y (for sigmoid + cross-entropy)
        dZ = cache[f'A{self.n_layers-1}'] - y
        
        # Backpropagate through layers
        for i in range(self.n_layers - 1, 0, -1):
            A_prev = cache[f'A{i-1}']
            
            # Gradients: dL/dW = (1/m) * A_prev^T * dZ, dL/db = (1/m) * sum(dZ)
            gradients[f'dW{i}'] = (1 / m) * np.dot(A_prev.T, dZ)
            gradients[f'db{i}'] = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
            
            if i > 1:
                # Propagate gradient to previous layer: dZ_prev = dZ * W^T * g'(Z_prev)
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self._relu_derivative(cache[f'Z{i-1}'])
        
        return gradients
    
    def _update_parameters(self, gradients):
        """Update weights and biases: W = W - lr * dW, b = b - lr * db"""
        for i in range(1, self.n_layers):
            self.weights[i] -= self.lr * gradients[f'dW{i}']
            self.biases[i] -= self.lr * gradients[f'db{i}']
    
    def fit(self, X, y):
        """Train the neural network"""
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        
        for iteration in range(self.n_iterations):
            # Forward propagation
            y_pred, cache = self._forward_propagation(X)
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward propagation
            gradients = self._backward_propagation(y, cache)
            
            # Update parameters
            self._update_parameters(gradients)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Return probability predictions"""
        y_pred, _ = self._forward_propagation(X)
        return y_pred
    
    def predict(self, X, threshold=0.5):
        """Return binary predictions"""
        return (self.predict_proba(X) >= threshold).astype(int)


# Test Case 1: XOR Problem (classic non-linear problem)
print("=" * 50)
print("Test Case 1: XOR Problem")
print("=" * 50)

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

model1 = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.5, n_iterations=2000)
model1.fit(X_xor, y_xor)

predictions = model1.predict(X_xor)
probs = model1.predict_proba(X_xor)
print(f"\nResults:")
for i, (x, true_y, pred_y, prob) in enumerate(zip(X_xor, y_xor, predictions.flatten(), probs.flatten())):
    print(f"  Input: {x}, True: {true_y}, Predicted: {pred_y}, Probability: {prob:.4f}")


# Test Case 2: Binary Classification with Hidden Patterns
print("\n" + "=" * 50)
print("Test Case 2: Binary Classification")
print("=" * 50)

np.random.seed(42)
# Create circular decision boundary
n_samples = 200
theta = np.random.uniform(0, 2 * np.pi, n_samples)
r = np.random.uniform(0, 3, n_samples)
X_circle = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y_circle = (r > 1.5).astype(int)

model2 = NeuralNetwork(layer_sizes=[2, 8, 4, 1], learning_rate=0.1, n_iterations=1000)
model2.fit(X_circle, y_circle)

predictions2 = model2.predict(X_circle)
accuracy = np.mean(predictions2.flatten() == y_circle)
print(f"\nTraining Accuracy: {accuracy:.4f}")


# Test Case 3: Linearly Separable Data (simpler case)
print("\n" + "=" * 50)
print("Test Case 3: Linearly Separable Data")
print("=" * 50)

np.random.seed(123)
X_class0 = np.random.randn(50, 2) + np.array([-2, -2])
X_class1 = np.random.randn(50, 2) + np.array([2, 2])
X_linear = np.vstack([X_class0, X_class1])
y_linear = np.hstack([np.zeros(50), np.ones(50)])

indices = np.random.permutation(100)
X_linear = X_linear[indices]
y_linear = y_linear[indices]

model3 = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.1, n_iterations=1000)
model3.fit(X_linear, y_linear)

predictions3 = model3.predict(X_linear)
accuracy3 = np.mean(predictions3.flatten() == y_linear)
print(f"\nTraining Accuracy: {accuracy3:.4f}")


# Test Case 4: Deep Network
print("\n" + "=" * 50)
print("Test Case 4: Deep Network (Multi-layer)")
print("=" * 50)

np.random.seed(456)
X_complex = np.random.randn(300, 3)
y_complex = ((X_complex[:, 0]**2 + X_complex[:, 1]**2) > 1).astype(int)

model4 = NeuralNetwork(layer_sizes=[3, 16, 8, 4, 1], learning_rate=0.05, n_iterations=1500)
model4.fit(X_complex, y_complex)

predictions4 = model4.predict(X_complex)
accuracy4 = np.mean(predictions4.flatten() == y_complex)
print(f"\nTraining Accuracy: {accuracy4:.4f}")
print(f"Network architecture: {model4.layer_sizes}")
print(f"Total parameters: {sum(w.size for w in model4.weights.values()) + sum(b.size for b in model4.biases.values())}")