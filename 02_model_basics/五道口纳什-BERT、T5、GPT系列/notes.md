> 之后分P

### 01 Tokenizer

1. Tokenizer 和 model 要匹配，tokenizer output -> model input
2. AutoTokenizer, AutorModel: 是 Generic Type 自适应找到模型

3. tokenizer 细节 [class PreTrainedTokenizerBase](https://github.com/huggingface/transformers/blob/7bb619d710ea3bcddeedb2e7999dff4e124aee85/src/transformers/tokenization_utils_base.py#L1389)

    3.1 len(input_ids) == len(attention_mask)，因为 mask 就是展示哪个 mask 了哪个没 mask 了，注意这里是 input_ids 不是文本长度
     
    3.2 tokenizer 调用时: [tokenizer.__call__()](https://github.com/huggingface/transformers/blob/7bb619d710ea3bcddeedb2e7999dff4e124aee85/src/transformers/tokenization_utils_base.py#L2897): encode, 具体是 \_\_call\_\_() -> _call_one() -> encode_plus()

    
    3.3 tokenizer.encode() 过程为: tokenizer.tokenize() + tokenizer.convert_tokens_to_ids()，具体来说还是在 [encode()](https://github.com/huggingface/transformers/blob/7bb619d710ea3bcddeedb2e7999dff4e124aee85/src/transformers/tokenization_utils_base.py#L2712) 中调用 [encode_plus()中的 _encoder_plus()](https://github.com/huggingface/transformers/blob/7bb619d710ea3bcddeedb2e7999dff4e124aee85/src/transformers/tokenization_utils_base.py#L3107)，具体实现在子类中。

    3.4 tokenizer.vocab 存储了 token -> id 的 mapping; special tokens: tokenizer.special_tokens_map

    3.5 attention mask 中为 1 的: 没有 mask 的，0的: mask的; 为什么为 0 ? -> 因为 [PAD] label 是 0. 用在 BERT MLM 预训练任务。

    3.6 token_type_ids 是 0 表示第一句，1表示第二句，可以通过 tokenizer.\_\_call\_\_，也能通过 encode_plus 返回。句子对一般是用在 Next Sentence Prediction 任务上（BERT 预训练）


> 文本数据集 Appendix 1




### 02 Model Architecture
以 BERT 为例

1. BERT 结构

    1.1 BERT 是 Encoder-Only 

    1.2 具体结构: embeddings + encoder(12 layer) + pooler

    - embeddings: token(word) ~, position ~, token_type ~

    - encoder: self attention (kqv) + feed forward

    1.3 参数量统计: total, learnable, 109000000

    - embedding: 21.7%

    - encoder: 77.6%

    - pooler: 0.5%

    1.4 BERT 和 BERTSequence: 后者加了一个 binary classification head

> 关于 no_grad 和 requires_grad=False 的区别， Appendix 2


2. BERT embedding 源码

    2.1 class BertModel: embeddings + encoder + pooler
    
    2.2 token + segment + position: 查表操作
    
    2.3 token: 3053 * 768, segment: 0/1 * 768, position: 512 * 768
    
    2.4 三种都是 ids -> 查表 -> 得到 embeddings
    
    - 使用 tokenizer() 可以得到 input_ids 和 token_type_ids
        
    - position_ids 就是顺序 12345这种
        
    2.5 embedding 获取： nn.Embedding() "查表"
    
    - token: model.embeddings.word_embeddings(input_ids)
        
    - segment: model.embeddings.token_type_embeddings(token_type_ids)
        
    - position: model.embeddings.position_embeddings(position_ids)
        
    - input: token + segment + position.unsqueeze(0) 这里也会自动扩展的 
        
    2.6 后处理
    
    - LayerNorm
        
    - dropout
        

> 关于 tokenizer 的分词细节 Appendix 3




3. [模型输出](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel.forward.returns)

    3.1  最少有两个 output: last_hidden_state 和 pooler_output，当 mdoel.config.output_hidden_states=True 时，另外输出 hidden_states.

    3.2  last_hidden_states: 

    - 最后一层的 hidden_states. 

    - 维度为：(batch_size, seq_len, hidden_size)

    3.3 pooler_output: 

    - 其实是两个过程的组合：首先提取出 last_hidden_layer 中 [CLS] token 对应的 embedding，然后通过一个全连接层(dense)，再加一个 tanh 激活函数之后的输出。 *所以说 pooler_output 是 [CLS] embedding 是错误的。*

    - 这个全连接层的参数是通过预训练时进行 next sentence prediction (classification)任务中的获得的。注意在 NSP 中会经历以下流程：

    | 步骤                | 输入维度          | 操作                          | 输出维度          |
    |---------------------|-------------------|------------------------------|-------------------|
    | [CLS]隐藏状态        | [batch_size, H]   | 无                            | [batch_size, H]   |
    | Pooler层（Linear层） | [batch_size, H]   | `W_pooler ∈ ℝ^{H×H}`          | [batch_size, H]   |
    | tanh激活             | [batch_size, H]   | 非线性变换                     | [batch_size, H]   |
    | NSP分类头（Linear层）| [batch_size, H]   | `W_nsp ∈ ℝ^{H×2}, b_nsp ∈ ℝ^2`| [batch_size, 2]   |

    - Pooler 层输出可以当作一个 **句子级别的 embedding 表示**，如果要对句子进行整体理解和表示，比如说分类和相似度匹配等操作，那么 Pooler output 就可以当作句子的 embedding。
    - 维度为 [batch_size, hidden_size]

    3.4 hidden_states:

    - 这是 BERT 每一层的 hidden_states

    - 维度为 (1 + layer_nums, batch_size, seq_length, hidden_size)，其中 layer_nums 对于 BERT-base 来说是 12层。多加的一层是 embedding 层的输出
        
    3.5 获取方式

    - last_hidden_states: output[0] == output[2][-1]

    - pooler_output: output[1] == model.pooler(output[2][-1]) 注意这里没有 *显式* 取出 [CLS]，因为这一步会在 BertPooler 中实现：

    ```python
    class BertPooler(nn.Module):
        def __init__(self, hidden_states):
            first_token_tensor = hidden_states[:, 0]  # 这里取出
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(first_token_tensor)
            return pooled_output
    ```

    - hidden_states: output[2] (如果只有 output_hidden_states=True)

> [关于 RoBERTa 移除 NSP 任务后的 Pooler 问题](https://www.zhihu.com/question/466862920)



4. Attention 计算

    4.1 关于 Attention 的维度推导：
    
    - 关于一个注意力头的推导：最开始我们有一个 embedding，这个 embedding 可以是上一层输出或者是输入的 embedding，记为 $E_{T\times d}$，其中 $T$ 是输入序列的长度，也就是 sequence length，$d$ 是 embedding 的维度。注意这里是一个注意力头的推导，所以没有 batch_size。另外我们有三个参数矩阵 $W_{k: d\times d_k}, W_{q: d\times d_q}, W_{v: d\times d_v}$。那么矩阵乘之后得到 $k_{T\times d_k}, q_{T\times d_q}, v_{T\times d_v}$。经过 softmax 得到注意力分数：
      
    $$attnScore_{T \times T} = softmax\left(\frac{q_{T\times d_q} \cdot k_{T\times d_k}^{T}}{\sqrt{d_q}}\right)$$
   
    所以得到的 attention score 就是 $T \times T$。并且一定有 $d_k = d_q$。最后乘以 $v_{T\times d_v}$ 就能得到 attention output $attn_{i}$，维度为 $T \times d_v$。所以说这里对于 $d_v$ 没有明确的要求。

    - 关于注意力的拼接：得到注意力结果之后拼接起来，也就是
   
    $$attn_{T\times (d_v \times n)} = \left(attn_{0}|attn_{1}|...|attn_{n - 1}\right)$$
   
    参数矩阵 KQV 也相似，对于每一个 $k$ 有 $k_i = E_{T\times d} \cdot W_{k_i: d\times d_k}$，把 $W_{k_i: d\times d_k}$ 拼接起来就有：
   
    $$K_{T\times (d_k \times n)} = \left(W_{k_0: d\times d_k} |  W_{k_1: d\times d_k} | ... | W_{k_{n - 1}: d\times d_k} \right)$$ 

    4.2 简单的 Attention 实现：
        - 一个简单的实现过程，不带 $W_O$：
   
        ```python
        ATTENTION_SIZE = 64  # ATTENTION_SIZE || embedding_size
    
        def get_K_or_Q_or_V(convert_matrix: Tensor, input: Tensor, bias: Tensor) -> Tensor:
            """Get K or Q or V in attention for one sample in a batch and on attention head
    
            Args:
                convert_matrix: [emb_size, emb_size], K or Q or V inself-attention layer
                input: [seq_len, emb_size], layer inputs
                bias: [emb_size], bias in self attention layer
            Returns:
                k or q or v: [seq_len, ATTENTION_SIZE]
            """
            # y = x @ A^T + b
            # ATTENTION_SIZE: size of each head in multi-head attention
            return input @ convert_matrix.weight.T[:, :ATTENTION_SIZE] + bias[:ATTENTION_SIZE]
    
    
        # 省略 input_tests embedding 得到的过程, model 初始化过程
        inputs = model.embeddings(
            input_tests['input_ids']，
            inputs_tests['token_type_ids']  # 这里 embeddings 层的输入输出要看一下
        )  # [batch_size, seq_len, emb_size]
    
    
        # 计算第一层 self-attention 的一个 attention head 的 attention score
        q = get_K_or_Q_or_V(
            convert_matrix=model.encoder.layer[0].attention.self.query.weight,
            input=inputs[0],  # 1st sample in batch
            bias=model.encoder.layer[0].attention.self.query.bias
        )  # [seq_len, ATTENTION_SIZE]
    
        k = get_K_or_Q_or_V(
            convert_matrix=model.encoder.layer[0].attention.self.key.weight,
            input=inputs[0],  # 1st sample in batch
            bias=model.encoder.layer[0].attention.self.key.bias
        )  # [seq_len, ATTENTION_SIZE]
    
        v = get_K_or_Q_or_V(
            convert_matrix=model.encoder.layer[0].attention.self.value.weight,
            input=inputs[0],  # 1st sample in batch
            bias=model.encoder.layer[0].attention.self.value.bias
        )  # [seq_len, ATTENTION_SIZE]
    
        attn_score = torch.nn.Softmax(dim=-1)(q @ k.T / math.sqrt(ATTENTION_SIZE))  # attention score, [seq_len, seq_len]
        attn_embed = attn_score @ v  # [seq_len, ATTENTION_SIZE], attention embedding
    
        # 拼接所有 attention attention embedding -> [seq_len, emb_size=ATTENTION_SIZE * num_head]
        ```

> [关于将输出映射到 embedding 维度的矩阵 W_O](https://www.cnblogs.com/rossiXYZ/p/18759167#%E7%9F%A9%E9%98%B5)


![trans-block](https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/trans_block.png)

5. 残差连接与残差模块

   5.1 在 attention 前后和 FFN 前后各有一次残差连接，见上图

   5.2 一个 block 的构成：

   ```shell
   (0): BertLayer(
        (attention): BertAttention(
            (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
            )
        )
        (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True) 
        )
        (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
        )
    )
   ```

   所以 (intermediate) 是 FFN 第一层，BertOutput (dense) 是第二层，维度从 768 -> 3072 -> 768。

   5.3 两次 add & norm:

   - 第一次 [BertSelfOutput](https://github.com/huggingface/transformers/blob/8f08318769c15fdb6b64418cbb070e8a8b405ffb/src/transformers/models/bert/modeling_bert.py#L425C1-L436C29)：
    
   ```python
   class BrosSelfOutput(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)  # 先 dropout 后 layerNorm
            hidden_states = self.LayerNorm(hidden_states + input_tensor)  # add & norm
            return hidden_states
   ```

   - 第二次 [BertOutput](https://github.com/huggingface/transformers/blob/8f08318769c15fdb6b64418cbb070e8a8b405ffb/src/transformers/models/bert/modeling_bert.py#L511)，和 BertSelfOutput 一样的流程
   


> [关于 BertIntermediate 层的必要性](https://zhuanlan.zhihu.com/p/552062991#:~:text=1.2%E3%80%81BertIntermediate%E5%8F%8A%E5%85%B6%E4%BD%9C%E7%94%A8%E8%AE%A8%E8%AE%BA)，以及附带 [paper](https://arxiv.org/abs/2012.11881)
>
> [关于在 BERTConig 里设置 BERTIntermediate 的 activation function 的 ISSUE](https://github.com/huggingface/transformers/issues/15)



### Appendix

1. 文本语料 newgroups: 媒体新闻
  
2. torch.no_grad versus requires_grad=False [可以看下 with torch.no_grad() 的源码]

    2.1 torch.no_grad

    - context manager，隐式地不进行梯度更新

    - 不改变模型内部的 requires_grad

    - 适用于 eval 阶段

    2.2 param.requires_grad

    - 显示的 freeze 一些 layer 的梯度更新

    - 在 layer 或者 module 级别进行 freeze

    - 可能会更加灵活一些 

    2.3 一个小例子：

    |torch.no_grad()|requires_grad=False|OOM|
    |--|--|--|
    |Y|Y|N|
    |Y|N|N|
    |N|Y|N|
    |N|N|Y|

    前两行因为 no_grad 阻止计算梯度，所以就没存梯度，第三行 requires_grad=False作用在整体模型，所以也没存，最后一个就 OOM 了

3. tokenizer 的分词细节

    3.1 使用 subword 和 wordpiece 的形式尽量将词利用现有词表完成映射，轻易不会将一个词处理为 [UNK] -> 例子

    3.2 使用 ## 表示拼接。有 5828 个词作为后缀的

    3.2 encoder: token -> ids; decoder: ids -> tokens -> word
