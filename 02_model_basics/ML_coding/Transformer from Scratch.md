**References**

- [Hwcoder Blog](https://hwcoder.top/Manual-Coding-3)
- [[llama.py|LLaMA2 impl]]


## 简单版实现

总体来说就是三个部分： Embedding, Encoder 和 Decoder。 简单版里边不涉及到:

- tokenizer

1. Embedding 里边分成 token embedding 和 position embedding。

```python
import torch
from torch import nn

class TokenEmbedding(nn.Module):
	def __init__(self, vocab_size):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_size)
	
	def forward(self, x):
		"""
		Args:
			x: input ids, (bt_size, seq_len)
		"""
		return self.embedding(x)  # (bt_size, seq_len, hidden_size)


class PostionEmbedding(nn.Module):
	pass

```




