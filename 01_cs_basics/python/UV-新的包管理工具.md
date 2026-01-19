---
source: https://zhuanlan.zhihu.com/p/1891901515553298344
---
安装 [`index-tts`](https://github.com/index-tts/index-tts?tab=readme-ov-file) 的时候遇到的新包管理工具，听说安装速度、解决包依赖问题比较nb，貌似是 23 年新推出的 Rust 版本的包管理工具，尝试一下新鲜事物

b 站这个 `tts` 里边说：

> [!IMPORTANT]
> We **only** support the `uv` installation method. Other tools, such as `conda`
> or `pip`, don't provide any guarantees that they will install the correct
> dependency versions. You will almost certainly have *random bugs, error messages,*
> ***missing GPU acceleration**, and various other problems* if you don't use `uv`.
> Please *do not report any issues* if you use non-standard installations, since
> almost all such issues are invalid.
> 
> Furthermore, `uv` is [up to 115x faster](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md)
> than `pip`, which is another *great* reason to embrace the new industry-standard
> for Python project management.

确实有点搞笑了。

好像都在用，问了一下GPT：

# 一句话总览

> **uv 用「项目为中心」的方式管理 Python 环境：  
> 每个项目 = 一个确定的 Python 版本 + 一个可复现的依赖快照 + 一个自动隔离的执行上下文**

---

## 一、uv 的核心设计思想（先讲哲学）

### 传统 Python 的问题

|方案|问题|
|---|---|
|`pip + venv`|人工步骤多，容易忘 activate|
|`conda`|环境重、慢、难复现|
|`pip install`|装了不等于对齐|
|`requirements.txt`|不能锁 transitive 依赖|

👉 **“环境”长期处于半失控状态**

---

### uv 的核心思想

uv 把环境当成：

> **一个可以被“完全描述 + 自动对齐”的状态**

所以 uv 只认三样东西：

1. **项目根目录**
    
2. **`pyproject.toml`**
    
3. **`uv.lock`**
    

---

## 二、uv 管理环境的 4 个核心组件

### 1️⃣ 项目作用域（Project-scoped environment）

uv **不搞全局环境**。

```
project/
├── pyproject.toml
├── uv.lock
└── .venv/   ← uv 专属环境
```

特点：

- `.venv` 默认就在项目里
    
- 不污染系统 Python
    
- 不共享（除非你强制）
    

👉 **环境和代码一一绑定**

---

### 2️⃣ Python 版本管理（隐式）

uv 会：

- 读取 `pyproject.toml` 里的：
    

```toml
[project]
requires-python = ">=3.10,<3.12"
```

- 或根据 lock 推断
    

然后：

- 使用你系统里已有的 Python
    
- 或提示你用 `uv python install 3.11`
    

📌 uv 不自己造 Python，但**会选对 Python**

---

### 3️⃣ 依赖管理 = 声明 + 锁定（核心）

#### pyproject.toml（声明）

```toml
dependencies = [
  "numpy>=1.24",
  "torch",
]
```

含义是：

> “我允许哪些版本”

---

#### uv.lock（锁定）

```text
numpy==1.26.4
torch==2.1.2
```

含义是：

> “我**实际**用的就是这些”

👉 **uv sync 只信 uv.lock**

---

### 4️⃣ 执行上下文管理（关键但容易被忽略）

这是 uv **最聪明的地方之一**。

当你运行：

```bash
uv run python script.py
```

uv 会：

1. 自动找到项目根目录
    
2. 找到对应 `.venv`
    
3. 临时修改：
    
    - `PATH`
        
    - `VIRTUAL_ENV`
        
4. 启动 python
    
5. 退出后恢复系统状态
    

👉 **你不需要 activate**

---

## 三、uv 是如何“保证干净”的？

### pip 的问题

```bash
pip install A
pip install B
```

结果：

- A 装了
    
- B 装了
    
- 但 A 的旧依赖还在
    

---

### uv 的 sync 模型

```bash
uv sync
```

内部逻辑是：

```text
当前环境 packages
        ↓
和 uv.lock 做 diff
        ↓
多的删
少的装
错的换
```

👉 **环境 = lock 文件的镜像**

---

## 四、uv vs 你熟悉的工具（核心差异）

### uv vs pip

|pip|uv|
|---|---|
|命令级|项目级|
|安装导向|状态导向|
|不会删除|会删除|
|慢|极快（Rust）|

---

### uv vs conda

|conda|uv|
|---|---|
|自带 Python|用系统 Python|
|环境重|环境轻|
|跨语言|专注 Python|
|慢|非常快|

---

## 五、一个完整生命周期（从零到运行）

```bash
git clone repo
cd repo
uv sync              # 建环境
uv run pytest        # 跑测试
uv run python main.py
```

你**从头到尾都没有碰 activate**。

---

## 六、你可以这样理解 uv（非常重要）

> **uv 不是“环境管理工具”  
> uv 是一个“环境状态管理器”**

它关心的不是：

> “我刚刚装了什么？”

而是：

> “现在的环境是不是 exactly 我描述的那样？”

---

## 七、给你一句「工程级心法」

> **把 `.venv` 当缓存  
> 把 `uv.lock` 当真理**

---

如果你愿意，下一步我可以：

- 画一张 **uv / pip / conda 的对比架构图**
    
- 讲 **uv.lock 和 poetry.lock 的差异**
    
- 深挖 **uv 的 resolver 为什么快**
    
- 或直接结合 **CS336 作业结构**讲为什么课程选 uv
    

你挑一个，我陪你深入下去。