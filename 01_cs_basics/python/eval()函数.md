`eval()` 函数的作用就是执行一个字符串表达式的值，并返回表达式的值；可以在读取一些文本文件转化为结构化的内容的时候使用，比如读 config 文件或者 txt 文件的时候把列表或者一些变量转换回来。


## 缘起

最近看到了在评估一些模型输出的时候，可以用作 str -> decimal 的转换工具，拿到 decimal 之后就能比较大小了，比如说：
```python
response = "1.1"
ground_truth = "1.5"  # recorded in json file

if eval(reponse) < eval(ground_truth):
    print(True)
else:
    print(False)
```
这种，源码在：[MedCalc](https://github.com/ncbi-nlp/MedCalc-Bench/blob/ee5c8d8d045de00174b6ba7647dd766c7a498858/evaluation/evaluate.py#L31)



## 功能

执行一些表达式：
```python
>>>x = 7
>>> eval( '3 * x' )
21
>>> eval('pow(2,2)')
4
>>> eval('2 + 2')
4
>>> n=81
>>> eval("n + 4")
85

# 执行简单的数学表达式
result = eval("2 + 3 * 4")
print(result)  # 输出: 14

# 执行变量引用
x = 10
result = eval("x + 5")
print(result)  # 输出: 15

# 在指定命名空间中执行表达式
namespace = {'a': 2, 'b': 3}
result = eval("a + b", namespace)
print(result)  # 输出: 5
```


或者简单一点，把字符串变量还原成原始的变量：
```python
>>> x='7'
>>> x
'7'
>>> a=eval(x)
>>> a
7
>>> b='a'     # 等价于b='7'
>>> c=eval(b)
>>> c
7
>>>
```

这里还能把字符串转为列表：
```python
zifu=" ['1709020230', '1707030416', '0', '0', '0']  "
print(type(zifu))
ls =eval(zifu)
print(type(ls))
```

输出就是：
```shell
<class 'str'>
<class 'list'>
```


甚至能提取用户输入值，而且还是多个：
```python
a,b=eval(input())
```
输入：10,5，得到 a=10，b=5



**References**

[1] https://www.runoob.com/python/python-func-eval.html
