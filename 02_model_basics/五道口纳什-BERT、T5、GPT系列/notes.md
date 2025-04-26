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
