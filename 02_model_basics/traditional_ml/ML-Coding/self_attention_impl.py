import numpy as np

class SelfAttention:
    """
    Single-head self-attention mechanism
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, d_model, d_k=None, d_v=None):
        """
        Args:
            d_model: input dimension
            d_k: key/query dimension (defaults to d_model)
            d_v: value dimension (defaults to d_model)
        """
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        
        # Initialize projection matrices
        self.W_q = np.random.randn(d_model, self.d_k) * 0.01
        self.W_k = np.random.randn(d_model, self.d_k) * 0.01
        self.W_v = np.random.randn(d_model, self.d_v) * 0.01
        
        # Cache for backward pass
        self.cache = {}
    
    def softmax(self, x, axis=-1):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, X, mask=None):
        """
        Forward pass of self-attention
        
        Args:
            X: input tensor of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
            mask: optional mask of shape (seq_len, seq_len) for masking attention
        Returns:
            output: attended values of shape (seq_len, d_v) or (batch_size, seq_len, d_v)
            attention_weights: attention weights of shape (seq_len, seq_len)
        """
        # Project to Q, K, V
        Q = X @ self.W_q  # (seq_len, d_k) or (batch, seq_len, d_k)
        K = X @ self.W_k  # (seq_len, d_k)
        V = X @ self.W_v  # (seq_len, d_v)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = Q @ K.T / np.sqrt(self.d_k)  # (seq_len, seq_len)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores, axis=-1)  # (seq_len, seq_len)
        
        # Apply attention to values
        output = attention_weights @ V  # (seq_len, d_v)
        
        # Cache for potential backward pass
        self.cache = {
            'X': X, 'Q': Q, 'K': K, 'V': V,
            'scores': scores,
            'attention_weights': attention_weights
        }
        
        return output, attention_weights


class MultiHeadAttention:
    """
    Multi-head attention mechanism
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: model dimension
            num_heads: number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head
        
        # Projection matrices for all heads (combined)
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01  # output projection
        
        self.cache = {}
    
    def softmax(self, x, axis=-1):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)
        Input shape: (seq_len, d_model)
        Output shape: (num_heads, seq_len, d_k)
        """
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)  # (num_heads, seq_len, d_k)
    
    def combine_heads(self, x):
        """
        Combine heads back to d_model
        Input shape: (num_heads, seq_len, d_k)
        Output shape: (seq_len, d_model)
        """
        x = x.transpose(1, 0, 2)  # (seq_len, num_heads, d_k)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.d_model)
    
    def forward(self, X, mask=None):
        """
        Forward pass of multi-head attention
        
        Args:
            X: input tensor of shape (seq_len, d_model)
            mask: optional mask of shape (seq_len, seq_len)
        Returns:
            output: attended values of shape (seq_len, d_model)
            attention_weights: attention weights of shape (num_heads, seq_len, seq_len)
        """
        seq_len = X.shape[0]
        
        # Linear projections
        Q = X @ self.W_q  # (seq_len, d_model)
        K = X @ self.W_k
        V = X @ self.W_v
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention for each head
        # Q @ K^T: (num_heads, seq_len, d_k) @ (num_heads, d_k, seq_len)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        # scores shape: (num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        # Softmax
        attention_weights = self.softmax(scores, axis=-1)
        # (num_heads, seq_len, seq_len)
        
        # Apply attention to values
        attended = np.matmul(attention_weights, V)
        # (num_heads, seq_len, d_k)
        
        # Combine heads
        output = self.combine_heads(attended)  # (seq_len, d_model)
        
        # Final linear projection
        output = output @ self.W_o  # (seq_len, d_model)
        
        self.cache = {
            'X': X, 'Q': Q, 'K': K, 'V': V,
            'scores': scores,
            'attention_weights': attention_weights
        }
        
        return output, attention_weights


# ============================================================
# Test Cases
# ============================================================

print("=" * 70)
print("Test Case 1: Single-Head Self-Attention")
print("=" * 70)

# Simple sequence: 3 tokens, each with 4 features
X = np.array([
    [1.0, 0.0, 0.0, 0.0],  # token 1
    [0.0, 1.0, 0.0, 0.0],  # token 2
    [0.0, 0.0, 1.0, 0.0]   # token 3
])

d_model = 4
attention = SelfAttention(d_model=d_model, d_k=4, d_v=4)

output, attn_weights = attention.forward(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"\nAttention weights (each row shows which tokens this token attends to):")
print(attn_weights)
print(f"\nAttention weights sum per row (should be 1.0): {attn_weights.sum(axis=1)}")
print(f"\nOutput (first 2 tokens):\n{output[:2]}")


print("\n" + "=" * 70)
print("Test Case 2: Self-Attention with Causal Mask")
print("=" * 70)

# Causal mask: token i can only attend to tokens <= i
seq_len = 4
causal_mask = np.tril(np.ones((seq_len, seq_len))).astype(bool)

X_causal = np.random.randn(seq_len, d_model)
attention_causal = SelfAttention(d_model=d_model)

output_causal, attn_weights_causal = attention_causal.forward(X_causal, mask=causal_mask)

print(f"Causal mask (1=attend, 0=mask):")
print(causal_mask.astype(int))
print(f"\nAttention weights with causal mask:")
print(attn_weights_causal.round(3))
print(f"\nNote: Each token only attends to current and previous tokens")


print("\n" + "=" * 70)
print("Test Case 3: Multi-Head Attention")
print("=" * 70)

d_model = 8
num_heads = 2
seq_len = 5

X_multi = np.random.randn(seq_len, d_model)
multi_head_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

output_multi, attn_weights_multi = multi_head_attn.forward(X_multi)

print(f"Input shape: {X_multi.shape}")
print(f"d_model: {d_model}, num_heads: {num_heads}, d_k per head: {d_model // num_heads}")
print(f"Output shape: {output_multi.shape}")
print(f"Attention weights shape: {attn_weights_multi.shape} (num_heads, seq_len, seq_len)")
print(f"\nHead 1 attention weights:")
print(attn_weights_multi[0].round(3))
print(f"\nHead 2 attention weights:")
print(attn_weights_multi[1].round(3))


print("\n" + "=" * 70)
print("Test Case 4: Attention on Sentence (Embeddings)")
print("=" * 70)

# Simulate word embeddings for sentence: "I love AI"
sentence_embeddings = np.array([
    [0.5, 0.2, 0.1, 0.8],   # "I"
    [0.3, 0.9, 0.2, 0.4],   # "love"
    [0.7, 0.1, 0.9, 0.3]    # "AI"
])

d_model = 4
attn_sentence = SelfAttention(d_model=d_model)
output_sentence, attn_weights_sentence = attn_sentence.forward(sentence_embeddings)

print(f"Sentence: ['I', 'love', 'AI']")
print(f"Input embeddings shape: {sentence_embeddings.shape}")
print(f"\nAttention weights (how much each word attends to others):")
print(f"         I      love    AI")
for i, word in enumerate(['I', 'love', 'AI']):
    print(f"{word:5s}: {attn_weights_sentence[i].round(3)}")

print(f"\nOutput contextual embeddings shape: {output_sentence.shape}")


print("\n" + "=" * 70)
print("Test Case 5: Multi-Head with Different Sequence Lengths")
print("=" * 70)

# Test with longer sequence
d_model = 16
num_heads = 4
seq_len = 10

X_long = np.random.randn(seq_len, d_model)
multi_head_long = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
output_long, attn_weights_long = multi_head_long.forward(X_long)

print(f"Input shape: {X_long.shape}")
print(f"Configuration: d_model={d_model}, num_heads={num_heads}, d_k={d_model//num_heads}")
print(f"Output shape: {output_long.shape}")
print(f"Attention weights shape: {attn_weights_long.shape}")
print(f"\nFirst head attention weights (first 5x5 submatrix):")
print(attn_weights_long[0, :5, :5].round(3))
