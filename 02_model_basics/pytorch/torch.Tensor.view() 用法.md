
### `torch.Tensor.view()` 用法简介

`view()` 用于重新调整张量的形状（不改变数据本身）。它返回一个新的张量，和原张量共享内存，因此修改一个会影响另一个。

#### 基本语法

```python
tensor.view(shape)
```

- `shape`：新的形状，可以是多个整数参数或一个元组。
    
- 其中某个维度可设为 `-1`，表示该维度由 PyTorch 自动推断。
    

#### 示例

```python
import torch

x = torch.arange(6)        # tensor([0, 1, 2, 3, 4, 5])
x = x.view(2, 3)           # 重新调整为 2x3
print(x)
# 输出：
# tensor([[0, 1, 2],
#         [3, 4, 5]])

y = torch.arange(8)
y = y.view(2, -1)          # 自动推断第二个维度为4
print(y)
# 输出：
# tensor([[0, 1, 2, 3],
#         [4, 5, 6, 7]])
```

#### 关于“连续的”张量

- **连续（contiguous）** 是指张量在内存中的数据是按照元素的顺序线性存放的。
    
- 某些操作（如 `transpose`、`permute`）只是改变数据的“视图”而不移动内存中的数据，此时张量会变成 **非连续的**。
    
- `view()` 需要底层数据是连续的，否则会报错。
    

例如：

```python
x = torch.arange(6).view(2, 3)
y = x.transpose(0, 1)          # 改变了维度顺序，变成非连续张量
print(y.is_contiguous())       # False

# 若想再使用 view()，需要先转为连续
z = y.contiguous().view(-1)
print(z)                       # tensor([0, 3, 1, 4, 2, 5])
```

> 简而言之：**非连续张量不能直接用 `view()`，需要先 `.contiguous()`。**



### `.view()` 与 `.reshape()` 的区别

|特性|`.view()`|`.reshape()`|
|---|---|---|
|内存要求|只能用于 **连续张量**|可以用于 **连续或非连续张量**|
|是否共享内存|若可能，则与原张量共享内存|尝试共享，但必要时会返回新拷贝|
|灵活性|较严格，常需要 `.contiguous()`|更灵活，能自动处理非连续情况|
|速度|略快（无需复制）|可能稍慢（若需要复制）|

#### 示例

```python
import torch

x = torch.arange(6).view(2, 3)
y = x.transpose(0, 1)        # 非连续张量

# view() 会报错
# y.view(-1)  # RuntimeError

# reshape() 能自动处理
z = y.reshape(-1)
print(z)  # tensor([0, 3, 1, 4, 2, 5])
```

**总结：**

- 如果你确定张量是连续的，用 `view()` 更高效。
    
- 如果不确定张量的连续性，用 `reshape()` 更安全。