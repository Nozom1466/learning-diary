
### `round()` 的舍入规则

Python 的 `round()` 使用 **银行家舍入法（Banker’s Rounding）**，即 **“四舍六入五取偶”**：

- 当小数部分不是 0.5 时，执行普通的四舍五入。
    
- 当小数部分正好为 0.5 时，舍入到**最接近的偶数**。
    

示例：

```python
round(4.5)  # 4
round(5.5)  # 6
round(6.5)  # 6
round(7.5)  # 8
```

此规则用于减少大规模统计或金融计算中的累计偏差。

若需传统意义上的“四舍五入”（0.5 一律进位），可使用：

```python
import math
def traditional_round(x):
    return math.floor(x + 0.5)
```
