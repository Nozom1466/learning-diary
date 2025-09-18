逐步调试 pdb 模块 [CSDN](https://blog.csdn.net/tekin_cn/article/details/145958203?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-5-145958203-blog-122582895.235^v43^pc_blog_bottom_relevance_base1&spm=1001.2101.3001.4242.4&utm_relevant_index=7)

1. 从命令行启动: 
```shell
python -m pdb your_script.py
```
然后等待你输入指令，常用指令：

| 命令        | 描述                                                                 |
|-------------|----------------------------------------------------------------------|
| n 或 next   | 执行下一行代码，不进入函数内部                                        |
| s 或 step   | 执行下一行代码，如果是函数调用，则进入函数内部                        |
| c 或 continue | 继续执行程序，直到遇到下一个断点                                    |
| b 或 break  | 设置断点，如 `b 10` 表示在第 10 行设置断点                           |
| cl 或 clear | 清除断点，如 `cl 10` 表示清除第 10 行的断点                          |
| p 或 print  | 打印变量的值，如 `p a` 表示打印变量 a 的值                           |
| l 或 list   | 列出当前代码段，默认显示当前行前后的 11 行代码                        |
| q 或 quit   | 退出调试器                                                            |

2. 程序中：添加 `import pdb; pdb.set_trace()`，在这个位置就直接暂停进入调试界面了

3. 条件断点：`b 10 if-condition` if-condition 就是满足这个条件的时候才在第 10 行加断点

4. 事后断点：`python -i your_script.py`, 当程序崩溃后，会进入交互式 Python 环境，此时可以输入 import pdb; pdb.pm() 来启动 pdb 调试，查看异常发生时的堆栈信息。
