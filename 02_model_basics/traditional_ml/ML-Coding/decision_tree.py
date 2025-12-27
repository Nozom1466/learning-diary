import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # feature index to split on
        self.threshold = threshold  # threshold value for split
        self.left = left           # left child node
        self.right = right         # right child node
        self.value = value         # leaf node prediction value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def _gini_impurity(self, y):
        """Gini impurity: 1 - sum(p_i^2)"""
        counter = Counter(y)
        impurity = 1.0
        total = len(y)
        for count in counter.values():
            prob = count / total
            impurity -= prob ** 2
        return impurity
    
    def _entropy(self, y):
        """Entropy: -sum(p_i * log2(p_i))"""
        counter = Counter(y)
        entropy = 0.0
        total = len(y)
        for count in counter.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def _information_gain(self, X_column, y, threshold):
        """Information gain: parent_entropy - weighted_avg(children_entropy)"""
        parent_entropy = self._entropy(y)
        
        # Split data
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Weighted average of children entropy
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        entropy_left = self._entropy(y[left_mask])
        entropy_right = self._entropy(y[right_mask])
        weighted_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        
        return parent_entropy - weighted_entropy
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        n_features_to_consider = self.n_features or n_features
        feature_indices = np.random.choice(n_features, n_features_to_consider, replace=False)
        
        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_child, right=right_child)
    
    def _most_common_label(self, y):
        """Return the most common class label"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def fit(self, X, y):
        """Build the decision tree"""
        self.root = self._build_tree(X, y)
        return self
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for a single sample"""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        """Make predictions for all samples"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def print_tree(self, node=None, depth=0):
        """Print tree structure for visualization"""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            print("  " * depth + f"Leaf: class={node.value}")
        else:
            print("  " * depth + f"X[{node.feature}] <= {node.threshold:.2f}")
            print("  " * depth + "Left:")
            self.print_tree(node.left, depth + 1)
            print("  " * depth + "Right:")
            self.print_tree(node.right, depth + 1)


# Test Case 1: Simple 2D classification
print("=" * 50)
print("Test Case 1: Simple 2D Classification")
print("=" * 50)

np.random.seed(42)
X_simple = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_simple = np.array([0, 0, 0, 1, 1, 1])

model1 = DecisionTree(max_depth=3, min_samples_split=2)
model1.fit(X_simple, y_simple)

predictions1 = model1.predict(X_simple)
accuracy1 = np.mean(predictions1 == y_simple)
print(f"Training Accuracy: {accuracy1:.4f}")
print(f"\nPredictions: {predictions1}")
print(f"True labels: {y_simple}")

print("\nTree Structure:")
model1.print_tree()


# Test Case 2: Iris-like dataset (3 classes)
print("\n" + "=" * 50)
print("Test Case 2: Multi-class Classification")
print("=" * 50)

np.random.seed(123)
# Generate 3-class data
X_class0 = np.random.randn(30, 2) + np.array([0, 0])
X_class1 = np.random.randn(30, 2) + np.array([3, 3])
X_class2 = np.random.randn(30, 2) + np.array([0, 3])

X_multi = np.vstack([X_class0, X_class1, X_class2])
y_multi = np.hstack([np.zeros(30), np.ones(30), np.full(30, 2)])

indices = np.random.permutation(90)
X_multi = X_multi[indices]
y_multi = y_multi[indices]

model2 = DecisionTree(max_depth=5, min_samples_split=5)
model2.fit(X_multi, y_multi)

predictions2 = model2.predict(X_multi)
accuracy2 = np.mean(predictions2 == y_multi)
print(f"Training Accuracy: {accuracy2:.4f}")

# Test on new data
X_test = np.array([[0, 0], [3, 3], [0, 3], [1.5, 1.5]])
predictions_test = model2.predict(X_test)
print(f"\nTest predictions for new samples:")
for i, (x, pred) in enumerate(zip(X_test, predictions_test)):
    print(f"  Sample {x}: Predicted class {int(pred)}")


# Test Case 3: XOR problem
print("\n" + "=" * 50)
print("Test Case 3: XOR Problem (Non-linear)")
print("=" * 50)

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                  [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
y_xor = np.array([0, 1, 1, 0, 0, 1, 1, 0])

model3 = DecisionTree(max_depth=3)
model3.fit(X_xor, y_xor)

predictions3 = model3.predict(X_xor)
accuracy3 = np.mean(predictions3 == y_xor)
print(f"Training Accuracy: {accuracy3:.4f}")
print(f"Predictions: {predictions3}")
print(f"True labels: {y_xor}")


# Test Case 4: Larger dataset with feature subset
print("\n" + "=" * 50)
print("Test Case 4: Larger Dataset with Random Features")
print("=" * 50)

np.random.seed(456)
n_samples = 200
n_features = 10

X_large = np.random.randn(n_samples, n_features)
# Create non-linear decision boundary using first 3 features
y_large = ((X_large[:, 0] > 0) & (X_large[:, 1] > 0)).astype(int)

# Use random subset of features (like Random Forest does)
model4 = DecisionTree(max_depth=6, min_samples_split=10, n_features=5)
model4.fit(X_large, y_large)

predictions4 = model4.predict(X_large)
accuracy4 = np.mean(predictions4 == y_large)
print(f"Training Accuracy: {accuracy4:.4f}")
print(f"Number of features: {n_features}, Features used per split: {model4.n_features}")


# Test Case 5: Overfitting demonstration
print("\n" + "=" * 50)
print("Test Case 5: Overfitting (Deep vs Shallow)")
print("=" * 50)

np.random.seed(789)
X_overfit = np.random.randn(50, 2)
y_overfit = (X_overfit[:, 0] > 0).astype(int)

# Deep tree (may overfit)
model5_deep = DecisionTree(max_depth=10, min_samples_split=2)
model5_deep.fit(X_overfit, y_overfit)
pred5_deep = model5_deep.predict(X_overfit)
acc5_deep = np.mean(pred5_deep == y_overfit)

# Shallow tree (more generalization)
model5_shallow = DecisionTree(max_depth=3, min_samples_split=10)
model5_shallow.fit(X_overfit, y_overfit)
pred5_shallow = model5_shallow.predict(X_overfit)
acc5_shallow = np.mean(pred5_shallow == y_overfit)

print(f"Deep tree (max_depth=10): Accuracy = {acc5_deep:.4f}")
print(f"Shallow tree (max_depth=3): Accuracy = {acc5_shallow:.4f}")
print("\nNote: Deep tree may achieve higher training accuracy but risks overfitting")