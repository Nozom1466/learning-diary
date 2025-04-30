## Introduction

所以 vllm 到底是啥？看官方 doc: "vLLM is a fast and easy-to-use library for LLM inference and serving."

**功能**: 主要是围绕着 PagedAttention 形成的一系列 model searving 相关的工具，包括 continuous batching, CUDA/HIP graph, Quantization, optimized CUDA kernel, speculative 
decoding .... 

**使用场景**: (other system?) integration with huggingface models, serving with various decoding algorithms, parallelism, prefix caching, multi-lora ....

**硬件支持**: NVIDIA, AMD, Intel CPU ....

> 除了 quantization, CUDA kernel 听说过以外其他的基本 0 了解。


**References**

[1] https://docs.vllm.ai/en/latest/



## PagedAttention

### 简单概述
**作者**：UCB [Sky Computing Lab](https://sky.cs.berkeley.edu/) 的项目

**效果**：
- PagedAttention 能做到 huggingface transformer 24 倍的吞吐量但是不需要改模型结构
- 相比 [HF]() 和 [TGI]() 相比 throughput 分别提升十倍、两三倍

**动机**:
- LLM serving 主要瓶颈就是在 GPU 里存 KV cache
- KV cache 1. 占的空间很大：LLaMA-13B 能有 1.7G. 2. 形状不定，取决于 sequence length

> [!TIP]
> 关于 KV cache 的动态大小: 
>
> [DeepSeek-R1] total KV cache size = N * 2 * L * H * d_head，N 是序列长度，L 是层数，H 是注意力头数，d_head 是每个头的维度. 2 是因为 K 和 V.


**简单原理**:

主要是在 GPU 上边更高效利用存储空间，让模型在计算 Attention 的时候更加高效。
通过类似操作系统分页管理的方法，PagedAttention 将连续的 K 和 V 存储在不连续的内存空间中，即连续的虚拟内存可以映射到非连续的物理内存中。
计算 Attention score 的时候，一次取出一组同时计算，加快计算效率。


![PagedAttention Memory](https://blog.vllm.ai/assets/figures/annimation0.gif)

类比操作系统的内容，blocks 类似 pages，tokens 类似 bytes，sequences 类似 processes. 比如以下的 KV 存储过程:

![Generation Process](https://blog.vllm.ai/assets/figures/annimation1.gif)


**性能分析**:
1. 空间浪费主要是在最后一个 block上，但是浪费率低于 4%，所以利用率很高了.
2. PagedAttention 存储方式考虑到了parallel sampling 并行生成的情况。这时候两个 sequence 都要生成，生成的内容不能写到一个区域，所以需要copy一下这一个 block 到其他位置之后，两个 sequence 继续生成，生成的结果分别写在copy前后两个地方，这样不会冲突。中间通过记录 reference count来检测是否将会发生写入冲突。具体看下边 gif。
3. 
![copy-on-write](https://blog.vllm.ai/assets/figures/annimation3.gif)

**References**

[1] https://blog.vllm.ai/2023/06/20/vllm.html



### PagedAttention 论文



**References**
[1] https://arxiv.org/abs/2309.06180



[TO BE CONTINUED]









