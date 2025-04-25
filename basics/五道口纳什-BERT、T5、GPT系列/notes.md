> 之后分P

### 01 Tokenizer

1. Tokenizer 和 model 要匹配，tokenizer output -> model input
2. AutoTokenizer, AutorModel: 是 Generic Type 自适应找到模型

最基本调包
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
batch_input = tokenizer(test_senteces, truncation=True, padding=True, return_tensors='pt')

print(model(**batch_input))
```

3. tokenizer 细节 [class PreTrainedTokenizerBase](https://github.com/huggingface/transformers/blob/7bb619d710ea3bcddeedb2e7999dff4e124aee85/src/transformers/tokenization_utils_base.py#L1389)

    3.1 len(input_ids) == len(attention_mask)，因为 mask 就是展示哪个 mask 了哪个没 mask 了，注意这里是 input_ids 不是文本长度
     
    3.2 tokenizer 调用时: tokenizer.__call__(): encode, 具体说是 [\_\_call\_\_()](https://github.com/huggingface/transformers/blob/7bb619d710ea3bcddeedb2e7999dff4e124aee85/src/transformers/tokenization_utils_base.py#L2897) -> [_call_one()](https://github.com/huggingface/transformers/blob/7bb619d710ea3bcddeedb2e7999dff4e124aee85/src/transformers/tokenization_utils_base.py#L3061) -> encode_plus()
    
    3.3 tokenizer.encode 过程为: tokenizer.tokenize + tokenizer.convert_tokens_to_ids

    3.4 tokenizer.vocab 存储了 token -> id 的 mapping; special tokens: tokenizer.special_tokens_map

    3.5 attention mask 中为 1 的: 没有 mask 的，0的: mask的; 为什么为 0 ? -> 因为 [PAD] label 是 0.

**Reference**
[1] https://github.com/chunhuizhang/bilibili_vlogs/blob/master/hugface/01_tokenizer_sentiment_analysis.ipynb
