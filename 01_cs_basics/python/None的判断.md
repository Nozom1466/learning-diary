空数组、空字典、空元组、空字符串和 None 在判断时的那些事


首先对于空列表、空元组、空字典、空字符串来说，取反 `not` 可以将其变为布尔类型，并且均为 `False` 。并且如果进行 `is None` 和 `== None` 的判断均为 `False`。比如说：
```python
empty_list = []
empty_tuple = ()
empty_dict = {}
empty_string = ""

print(empty_list, type(empty_list))  # [] <class 'list'>
print(empty_tuple, type(empty_tuple))  # () <class 'tuple'>
print(empty_dict, type(empty_dict))  # {} <class 'dict'>
print(empty_string, type(empty_string))  # "" <class 'str'>


print(not empty_list, type(not empty_list))  # True <class 'bool'>
print(not empty_tuple, type(not empty_tuple))  # True <class 'bool'>
print(not empty_dict, type(not empty_dict))  # True <class 'bool'>
print(not empty_string, type(not empty_string))  # True <class 'bool'>


print(empty_list is None)  # False
print(empty_tuple is None)  # False
print(empty_dict is None)  # False
print(empty_string is None)  # False


print(empty_list == None)  # False
print(empty_tuple == None)  # False
print(empty_dict == None)  # False
print(empty_string == None)  # False
```

对于 `None` 来说，取反之后是 `True`，但却是 `NoneType`([在线网站测试](https://www.online-python.com/))：
```python
empty = None

print(empty, type(empty))  # None <class 'NoneType'>
print(not empty, type(empty))  # True <class 'NoneType'>
print(empty is None)  # True
print(empty == None)  # True
```


在判断某个变量是不是 `None` 时，不要使用 `not var` 的形式判断，因为这时无论是不是 `None` 经过 `not` 之后都变成 `True` 了，可以使用 `is` 相关的方法。比如说：
```python
x = []
print(not x)  # True
print(x is None)  # False
print(not x is None)  # True   # 这里 is 比 not 优先级高

x = None
print(not x)  # True
print(x is None)  # True
print(not x is None)  # False
```







**References**

[1] https://blog.csdn.net/qq_36171491/article/details/124728283, 存疑

[2] https://www.online-python.com/

[3] https://blog.csdn.net/wbiblem/article/details/72885596
