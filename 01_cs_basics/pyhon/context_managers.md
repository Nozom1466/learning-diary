> 遇到 `with torch.no_grad():`，顺便看下 context manager

### context manager 是什么 ?

在 `with open('test.txt') as f:` 中 `open('test.txt')` 就是上下文管理器，`f` 是资源对象，也就是下边的形式：

```python
with EXPR as VAR:
    BLOCK
```


### context manager 结构是什么 ?
 
主要是实现 `__enter__` 和 `__exit__` 方法，比如：

```python
class Resource():
    # 1
    def __enter__(self):
        print('===connect to resource===')
        return self

    # 2
    def operate(self):
        print('===in operation===')

    # 3
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('===close resource connection===')
        
if __name__ == '__main__':
    with Resource() as res:
        res.operate()
```
按照 #1 #2 #3 的顺序执行，过程就是连接资源、执行操作、关闭连接。



### 为什么要用 context manager ?

优雅操作资源，更好复用：文件以及数据库连接，以及*处理异常*。 `with` 可以在 `__exit__` 中进行异常处理，从而在主代码中将异常处理隐藏起来。对于以下代码，异常在 `__exit__` 中通过异常类型(exc_type), 异常值(exc_val), 异常的错误栈信息(exc_tb) 捕获到了异常，最后返回了 True 表示就不需要抛出异常了。这样提高了代码的可读性，不用 `try` `except` 一堆了。

```python
class Resource():
    def __enter__(self):
        print('===connect to resource===')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('===close resource connection===')
        return True

    def operate(self):
        1/0

if __name__ == '__main__':
    with Resource() as res:
        res.operate()
```


### context manager 实现方式
可以使用类，也可以使用 `contextlib` 装饰器。如以下示例实现了自定义的上下文管理器以及异常处理。注意中间的 `yield` 出的就是资源对象。这里实际的调用过程是：遇到 `yield` -> 生成 generator -> 装饰器 `contextmanager` 用 `GeneratorContextManager` 包住 generator，最后给出的是 `GeneratorContextManager` 对象，这也是我们能够使用 `f.write()` 的原因，实际上是在调用 `GeneratorContextManager`。 

```python
import contextlib

@contextlib.contextmanager
def open_func(file_name):
    # __enter__方法
    print('open file:', file_name, 'in __enter__')
    file_handler = open(file_name, 'r')

    try:
        yield file_handler
    except Exception as exc:
        # deal with exception
        print('the exception was thrown')
    finally:
        print('close file:', file_name, 'in __exit__')
        file_handler.close()

        return

if __name__ == '__main__':
    with open_func('/Users/MING/mytest.txt') as file_in:
        for line in file_in:
            1/0
            print(line)

```


**References**

[1] https://www.cnblogs.com/wongbingming/p/10519553.html

[2] https://book.pythontips.com/en/latest/context_managers.html