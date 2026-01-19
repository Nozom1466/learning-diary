
## GPU Programming


### Getting started with Kernel

首先以编写一个 `mapKernel` 为例，输入是 原来的 `tensor`，以及输入的 `tensor` 的 `shape` 之类的信息，以及输出的 `tensor` 的 `shape` 信息，输出是经过 `map` 操作之后的 `tensor`。

如果以简单流程来说，假设输入的 `tensor` 的维度是 `2 * 3`，那么其实真正的输入数组是一个一维数组 `1 * 6`，所以会有位置映射的操作，从二维到一维，`map` 操作，然后一维到二维找到输出的位置。

这中间涉及到一个 `broadcast` 步骤，这步骤解决的问题是：如果输出的 `tensor` 的维度是 `[2， 3]`，但是输入的维度却是 `[1, 3]`，这样就会出现一个问题，对于 `out[1][0]` 这个元素，他去找输入中的 `[1][0]` 位置的时候找不到，因为输入就只有一行，所以这时候需要 `broadcast` 一下，我们去找 `in[0][0]`，也就是去第一行找。所以这个过程就是：我知道了输出我应该输出到什么位置，但是我去输入里找对应位置的时候找不到了，那就利用 `broadcast` 去第一行找（对于 2 dim 来说）。


### nvcc compilation

```bash
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
```

**参数说明**

| 参数 | 含义 |
|------|------|
| `nvcc` | NVIDIA CUDA 编译器 |
| `-o <path>` | 指定输出文件路径 |
| `--shared` | 编译成共享库（.so），而非可执行文件 |
| `src/combine.cu` | CUDA 源代码文件 |
| `-Xcompiler` | 将后续参数传递给 C++ 编译器 |
| `-fPIC` | Position Independent Code，生成位置无关代码（动态库必需） |

**编译流程**

```
.cu 源码 → nvcc 编译 → .so 共享库 → Python ctypes 加载 → 调用 CUDA 函数
```

**注意事项**

- `.so` = Shared Object，Linux 下的动态链接库
- 每次修改 `.cu` 文件后都需要重新编译
- `-fPIC` 是生成动态库的必要参数，否则无法被正确加载



## GPU Programming Basics

这个 GPU 编程非常头疼，主要原因是涉及到 thread 并行以及不了解 cuda 运行的主要流程。

代码参考:  [llmsys assignment 1 official codebase](https://github.com/llmsystem/llmsys_hw1)

### 映射关系：*一维存储数组* 与 *高维概念数组*  的映射关系

简单来说，在 kernel 看来你所谓的二维、三维等等数组其实全都以 **1维** 的形式来存储。也就是说在 kernel 中，你需要手动进行 *一维存储数组* 与 *高维概念数组* 之间的映射。比如说，一个二维数组的映射关系：

```
2 dim tensor: [2, 3]
[[1, 2, 3], 
[4, 5, 6]]

stored as 1 dim tensor: [6]
- stride = [3, 2]: [1, 2, 3, 4, 5, 6] <- kernel 实际看到的
- stride = [2, 1]: [1, 4, 2, 5, 3, 6]
```

其中多种存储方式主要是取决于 **stride** 的设置。你可以通过使用 stride 快速完成从*一维存储数组* 到 *高维概念数组* 的映射。stride 的一个语义含义就是，你想要在 *一维存储数组* 当中访问 *高维概念数组*  的在这个维度的下一个元素，需要下标 + 多少 ，比如说在 `[2, 1]` 中，2 这个元素要在 1 这个维度上访问 5，需要 +1，而这个 1 就是 `stride[1]`。具体方式：

```c++
__device__ void to_index(int ordinal, const int *shape, int *out_index, int num_dims)
{
  /**
   * Convert an ordinal to an index in the shape. Should ensure that enumerating position 0 ... size of
   * a tensor produces every index exactly once. It may not be the inverse of index_to_position.
   * Args:
   *    ordinal: ordinal position to convert
   *    shape: tensor shape
   *    out_index: return index corresponding to position
   *    num_dims: number of dimensions in the tensor
   *
   * Returns:
   *    None (Fills in out_index)
   */
  int cur_ord = ordinal;
  for (int i = num_dims - 1; i >= 0; --i)
  {
    int sh = shape[i];
    out_index[i] = cur_ord % sh;
    cur_ord /= sh;
  }
}
```

以上是从 *一维存储数组* 的下标 映射到 *高维概念数组* 的坐标，反过来也是一样通过 stride 计算下标即可：

```c++
__device__ int index_to_position(const int *index, const int *strides, int num_dims)
{
  /**
   * Converts a multidimensional tensor index into a single-dimensional position in storage
   * based on strides.
   * Args:
   *    index: index tuple of ints
   *    strides: tensor strides
   *    num_dims: number of dimensions in the tensor, e.g. shape/strides of [2, 3, 4] has 3 dimensions
   *
   * Returns:
   *    int - position in storage
   */
  int position = 0;
  for (int i = 0; i < num_dims; ++i)
  {
    position += index[i] * strides[i];
  }
  return position;
}
```



### Broadcasting 

思路就是：我输出算作比较大的数组，为了计算某一个输出的元素，我需要到输入数组中找到对应位置来进行计算。但是由于输入数组的形状比较小，而你要找的那个元素在输入数组中的位置不存在，这时候你就需要去找这个维度上，需要最小的那个元素作为你的输入，这个流程就是广播。例子：

```
A = [[1, 2, 3],        # shape: (2, 3)
     [4, 5, 6]]

B = [10, 20, 30]       # shape: (3,)  → 广播成 (2, 3)

A + B = [[11, 22, 33],
         [14, 25, 36]]
```

比如说 `A+B` 的 `[1, 2]` 这个位置，对应到 `A` 和 `B` 中应该是 `[1, 2]` 这个位置，`A` 中有这个位置，`B` 中没有，所以 `B` 应该去找 `B[0, 2] = 20`, 从结果上看就是 `B` 沿着第 0 个维度复制了一下，实际上是通过访存下标变化来实现。具体代码：

```c++
__device__ void broadcast_index(const int *big_index, const int *big_shape, const int *shape, int *out_index, int num_dims_big, int num_dims)
{
  /**
   * Convert a big_index into big_shape to a smaller out_index into shape following broadcasting rules.
   * In this case it may be larger or with more dimensions than the shape given.
   * Additional dimensions may need to be mapped to 0 or removed.
   *
   * Args:
   *    big_index: multidimensional index of bigger tensor
   *    big_shape: tensor shape of bigger tensor
   *    shape: tensor shape of smaller tensor
   *    nums_big_dims: number of dimensions in bigger tensor
   *    out_index: multidimensional index of smaller tensor
   *    nums_big_dims: number of dimensions in bigger tensor
   *    num_dims: number of dimensions in smaller tensor
   *
   * Returns:
   *    None (Fills in out_index)
   */
  for (int i = 0; i < num_dims; ++i)
  {
    if (shape[i] > 1)
    {
      out_index[i] = big_index[i + (num_dims_big - num_dims)];
    }
    else
    {
      out_index[i] = 0;
    }
  }
}
```

### GPU 的结构管理




### 一些拓展资料

- [cuda 13.0 以前的 guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#) 
- [cuda 13.0 以后的 guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Tutorial: OpenCL SGEMM tuning for Kepler](https://cnugteren.github.io/tutorial/pages/page1.html)

> cuda 现在转向 cuTile 编程，好像类似于 `triton` 的想法，把 thread 底层的调用封装起来了
