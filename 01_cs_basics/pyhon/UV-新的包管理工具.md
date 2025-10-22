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