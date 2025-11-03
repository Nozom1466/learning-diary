
*nanochat decomposition

主类 GPT

forward: 
- embedding + norm + embedding + norm + lm_head
- if training: cross entropy loss
- if inference: return logits
- both add softcap, use tanh


generate:
- for in range(max_tokens) loop
- get the last  logits
- use topk sampling -> set others to 0 or greedy sampling
- temperatures 
- softmax prob
- multinomial sampling

For 1 transformer block:

forward:
- norm
- attn -> kv_cache
- residual conn
- norm  -> prenorm
- mlp
- residual

For MLP:

forward: 
- full connection layer
- relu  ->  add square?
- projection layer


For attention layer:

forward:
- 输入就是: bs, seq, n_emb
- 用 nn.Linear 代替矩阵乘法
- Linear 出来的大小是 bs, seq, nhead \* head_dim
- 然后 view 变成 bs, seq, numheads, dim
	- 这里注意 q 是 numheads, kv 是 num_kv head -> GQA 和 MQA, MHA区别
- 然后进行的是相对位置编码RoPE？在计算完 Q和 K 之后。
- QK norm
- insert KV
- SScale dot  计算 attention score，是否用 gqa
	- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
	- 这里有个 enable_gqa 控制开关：https://chatgpt.com/share/690908c2-987c-8012-b984-c25cff10d206，他这里实际上还是扩展+计算
	- 这里的 还要区分一下训练和推理和chunk推理


RoPE:
