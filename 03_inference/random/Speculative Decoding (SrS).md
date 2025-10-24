---
source1: https://arxiv.org/pdf/2302.01318
improve1: https://arxiv.org/pdf/2409.10644?
---

算法：

```pseudo
Algorithm 2: Speculative Sampling (SpS)
Inputs:
    K      ← lookahead length
    T      ← minimum target sequence length
    q(·|·) ← auto-regressive target model
    p(·|·) ← auto-regressive draft model
    x₀, …, x_t ← initial prompt

Initialize:
    n ← t

while n < T do
    # Step 1: Draft sampling
    for t = 1 to K do
        sample 𝑥̃_t ~ p(x | x₁, …, x_n, 𝑥̃₁, …, 𝑥̃_{t−1})
    end for

    # Step 2: Compute target logits (in parallel)
    compute q(x | x₁, …, x_n),
            q(x | x₁, …, x_n, 𝑥̃₁),
            …,
            q(x | x₁, …, x_n, 𝑥̃₁, …, 𝑥̃_K)

    # Step 3: Acceptance / rejection sampling
    for t = 1 to K do
        sample r ~ Uniform[0, 1]

        if r < min(1, q(𝑥̃_t | x₁, …, x_{n+t−1}) / p(𝑥̃_t | x₁, …, x_{n+t−1})) then
            x_{n+t} ← 𝑥̃_t
            n ← n + 1
        else
            sample x_{n+t} ~ [q(x | x₁, …, x_{n+t−1}) − p(x | x₁, …, x_{n+t−1})]_+
            break   # exit for-loop
        end if
    end for

    # Step 4: If all K tokens were accepted
    if all K tokens accepted then
        sample x_{n+K+1} ~ q(x | x₁, …, x_n, x_{n+K})
        n ← n + 1
    end if
end while
```


QA:

整体思想是什么？和自回归解码有什么区别？
	自回归伪代码：
	
	```pseudo
	Algorithm 1: Auto-regressive (ArS) with Auto-Regressive Models
	
	Given auto-regressive target model q(·|·) and initial prompt sequence x₁, …, xₜ
	and target sequence length T.
	
	Initialise n ← t
	
	while n < T do
	    Sample xₙ₊₁ ~ q(x | x₁, …, xₙ)
	    n ← n + 1
	end while
	```

【论文】论文里边介绍了三个主要的时间消耗来源有哪些？
	Linear, Attn, All-reduce

介绍一下算法中的 rejection sampling

为什么全部 K 个 token accept 之后，还要再多采样一个？

能不能简单实现一下啊啊啊？