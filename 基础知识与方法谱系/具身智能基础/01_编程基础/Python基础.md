# Python 基础：从语法到 AI Research Code

Python 是许多 AI、计算机视觉、机器人学习和 World Model 项目的主要编程语言。研究代码中的 train.py、dataset.py、model.py、config.py，通常都建立在 Python 的变量、函数、模块、类、异常和数据结构之上。

本篇的目标不是让你背完 Python 语言，而是建立一条能通向科研代码的理解路径：

```text
看懂变量和数据类型
    ↓
看懂列表、字典和循环
    ↓
看懂函数和参数
    ↓
看懂模块、import 和项目目录
    ↓
看懂 class、继承和对象
    ↓
看懂异常、配置和工程入口
    ↓
开始阅读 Dataset、nn.Module 和训练脚本
```

> [!NOTE]
> 本篇负责 Python 语言与工程基础。Python 解释器和环境见 [[Conda与环境管理]]；命令行、路径和 Shell 见 [[Linux与Shell]]；版本管理见 [[Git]]；通用排错见 [[Debug与工程基础]]；数组基础见 [[NumPy]]；Tensor、Dataset、DataLoader 和 nn.Module 见 [[PyTorch]]。

## 0. 学习目标

学完以后，应该能够：

- 解释变量、对象、类型、表达式和赋值。
- 读懂数字、字符串、布尔值、列表、元组、字典和集合。
- 使用索引、切片、条件判断和循环处理数据。
- 读懂函数定义、参数、返回值和作用域。
- 理解模块、import、包和 Python 项目目录。
- 读懂 class、对象、属性、方法和继承的基本代码。
- 使用异常、assert 和基本文件操作进行工程排查。
- 理解 if __name__ == "__main__": 这类入口代码。
- 看懂 AI 项目中 dataset、model、config 和 train 几类文件的基本关系。
- 能够把 Python 代码拆成“输入 → 处理 → 输出”来阅读。

## 1. 为什么先学 Python？

AI 研究代码常把复杂任务拆成多个对象和函数：

```text
train.py
├── 读取参数和配置
├── 创建 Dataset / DataLoader
├── 创建 Model
├── 创建 Optimizer
├── 循环训练
└── 保存 checkpoint

dataset.py
└── 读取和转换样本

model.py
└── 定义模型类和 forward

utils.py
└── 提供可复用函数
```

如果只会逐行看语法，却不理解函数、对象和模块之间的关系，就会觉得研究代码“每一行都认识，但整体看不懂”。

Python 对 AI 代码的价值主要在于：

- 用变量保存数据和配置。
- 用列表、字典组织样本、参数和结果。
- 用函数封装重复操作。
- 用类表示 Dataset、模型、训练器等对象。
- 用模块把大型项目拆成多个文件。
- 用异常和日志处理真实运行中的问题。

## 2. 核心概念

### 2.1 对象（Object）

Python 中几乎所有运行时数据都可以看成对象。对象有：

- 值或内容。
- 类型。
- 可调用的方法或可访问的属性。

```python
x = 3
name = "robot"
items = [1, 2, 3]

print(type(x))
print(type(name))
print(type(items))
```

x、name、items 是变量名；3、robot 字符串和列表是对象；int、str、list 是类型。

### 2.2 变量与赋值

变量名是指向对象的标签，不是一个只能装某种类型的固定盒子。

```python
x = 3
x = "three"
```

这段代码可以运行，但 x 先指向整数，后来指向字符串。Python 是动态类型语言：变量不需要提前声明固定类型，但程序运行时对象仍然有明确类型。

赋值不是数学等式：

```python
a = 1
b = a
a = 2
print(b)
```

b 仍然是 1，因为执行 b = a 时，b 指向当时的对象。对可变对象需要更谨慎，后文会解释。

### 2.3 表达式与语句

**表达式**可以计算出一个值：

```python
2 + 3
len("robot")
x > 0
```

**语句**执行一个动作：

```python
x = 2 + 3
if x > 0:
    print(x)
```

科研代码中常见的代码块由缩进表示。Python 不使用大括号表示代码块；同一层级的缩进必须一致。

### 2.4 类型（Type）

常见基础类型：

| 类型 | 例子 | 用途 |
|---|---|---|
| int | 3 | 整数、索引、计数 |
| float | 0.001 | 连续数值、学习率 |
| str | "train" | 文本、路径片段 |
| bool | True/False | 条件 |
| NoneType | None | 表示没有值或缺省 |
| list | [1, 2] | 有序、可修改集合 |
| tuple | (1, 2) | 有序、通常不修改 |
| dict | {"lr": 0.001} | 键值映射 |
| set | {"train", "val"} | 不重复集合 |

类型影响可以执行的操作。看到错误时，先用 type(x) 和 repr(x) 检查对象，见 [[Debug与工程基础]]。

## 3. 基本数据类型与运算

### 3.1 数字

```python
a = 7
b = 2

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a // b)
print(a % b)
print(a ** b)
```

/ 返回浮点除法；// 是整除；% 是余数；** 是幂。训练代码中的学习率、epoch、batch size 和维度经常是数字，但它们的语义不同，不能只看类型。

### 3.2 字符串

```python
name = "world model"
message = f"training {name}"
print(message)
print(name.upper())
print(name.split())
```

f-string 可以把变量放入文本。科研代码中常用它生成日志、输出目录和错误提示。

```python
epoch = 3
loss = 0.42
print(f"epoch={epoch}, loss={loss:.4f}")
```

字符串不可变：不能直接修改其中某一个字符，通常需要生成新字符串。

### 3.3 布尔值与比较

```python
x = 5

print(x > 0)
print(x == 5)
print(x != 3)
print(0 < x < 10)
```

注意 = 是赋值，== 是比较。条件表达式最终通常得到 True 或 False。

逻辑运算：

```python
is_valid = x > 0 and x < 10
is_empty = x == 0 or x is None
not_ready = not is_valid
```

判断 None 时推荐使用 is None，而不是把 None 当普通值比较。

## 4. 列表、元组、字典和集合

### 4.1 列表（list）

列表是有序、可修改的对象集合。

```python
samples = ["image_1.png", "image_2.png"]
samples.append("image_3.png")
print(samples[0])
print(len(samples))
```

列表常用于样本路径、loss 历史、模块列表和 batch 中的普通 Python 数据。

切片：

```python
print(samples[:2])
print(samples[-1])
```

切片通常返回一个新列表。列表中也可以嵌套列表，但 NumPy 数组更适合规则数值计算，见 [[NumPy]]。

### 4.2 元组（tuple）

元组有序，通常表示不希望修改的一组值。

```python
image_size = (224, 224)
height, width = image_size
```

函数可以返回多个值，实际通常打包成元组：

```python
def get_size():
    return 224, 224

h, w = get_size()
```

元组解包在 AI 代码中很常见，例如 batch、shape 和函数返回值。

### 4.3 字典（dict）

字典保存键和值的映射。

```python
config = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "device": "cuda",
}

print(config["batch_size"])
config["epochs"] = 100
```

AI 项目中的配置、样本字段、模型输出和统计结果常使用字典。

更安全的读取：

```python
seed = config.get("seed", 0)
```

config["seed"] 不存在会触发 KeyError；get 可以提供默认值，但不要用默认值掩盖本应报错的配置缺失。

遍历：

```python
for key, value in config.items():
    print(key, value)
```

### 4.4 集合（set）

集合不保存重复元素，适合检查成员关系。

```python
splits = {"train", "val", "test"}
print("train" in splits)
```

集合没有稳定的顺序。需要有序数据时不要用 set 代替 list。

## 5. 索引、切片与可变对象

### 5.1 索引从 0 开始

```python
items = ["a", "b", "c"]
print(items[0])
print(items[-1])
```

第一个元素索引是 0，最后一个元素可以用 -1。NumPy 和许多深度学习 API 也遵循这一规则。

### 5.2 可变与不可变

list 和 dict 通常是可变对象；int、float、str 和 tuple 通常不可变。

```python
a = [1, 2]
b = a
b.append(3)
print(a)
```

a 也会变，因为 a 和 b 指向同一个列表对象。

复制：

```python
a = [1, 2]
b = a.copy()
b.append(3)
print(a)
print(b)
```

嵌套结构需要考虑深拷贝，但当前阶段知道“赋值可能共享同一个可变对象”即可。Debug 时看到列表或配置被意外改变，要检查对象引用。

## 6. 条件与循环

### 6.1 if / elif / else

```python
loss = 0.2

if loss < 0:
    status = "invalid"
elif loss < 1:
    status = "normal"
else:
    status = "large"
```

条件分支常用于设备选择、训练模式、配置开关和错误处理。

### 6.2 for 循环

```python
for epoch in range(3):
    print(epoch)
```

range(3) 产生 0、1、2，不包含 3。训练代码常见：

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
```

这里外层循环是 epoch，内层循环是 batch。完整 PyTorch 训练见 [[PyTorch]]。

### 6.3 enumerate

需要同时得到索引和值时：

```python
for index, name in enumerate(["train", "val"]):
    print(index, name)
```

### 6.4 zip

需要并行遍历多个序列时：

```python
names = ["a", "b"]
scores = [0.8, 0.9]

for name, score in zip(names, scores):
    print(name, score)
```

zip 通常按较短序列停止。输入长度不一致时要先确认这是否符合预期。

### 6.5 while

```python
count = 0
while count < 3:
    print(count)
    count += 1
```

while 依赖条件最终变为 False；忘记更新条件可能造成无限循环。训练和数据处理通常更常用 for。

## 7. 函数：封装可复用逻辑

### 7.1 定义和调用

函数是给一段操作命名，以便重复调用。

```python
def normalize_value(x, maximum):
    return x / maximum

result = normalize_value(128, 255)
```

函数包含：

- def：开始定义。
- 函数名：描述意图。
- 参数：调用者提供的输入。
- 函数体：要执行的逻辑。
- return：返回结果。

### 7.2 参数与默认值

```python
def make_output_dir(root, name="default"):
    return root / name
```

有默认值时可以省略对应参数。默认参数应该是稳定、明确的值；不要把可变列表作为默认参数：

```python
def bad(item, history=[]):
    history.append(item)
    return history
```

这会在多次调用间共享同一个 history。更安全：

```python
def good(item, history=None):
    if history is None:
        history = []
    history.append(item)
    return history
```

### 7.3 位置参数与关键字参数

```python
def train(model, data, epochs=10):
    pass

train(model, loader, 20)
train(model=model, data=loader, epochs=20)
```

关键字参数更清楚，尤其在训练脚本有多个同类型参数时。

### 7.4 返回值与 None

```python
def log_message(message):
    print(message)

result = log_message("start")
print(result)
```

没有显式 return 的函数返回 None。看到后续代码对 None 调用方法，要向前检查函数是否忘记返回值，见 [[Debug与工程基础]]。

### 7.5 作用域

函数内部变量通常是局部变量：

```python
def build_name():
    name = "model"
    return name

name = build_name()
```

不要过度依赖全局变量。研究代码中把配置、模型和数据作为参数传入函数，通常更容易测试和 Debug。

### 7.6 类型提示

类型提示帮助人和工具理解输入输出：

```python
def add(a: int, b: int) -> int:
    return a + b
```

类型提示通常不会自动阻止运行时传入错误类型，但能提高可读性和编辑器检查能力。

## 8. 模块、import 与项目结构

### 8.1 模块是什么？

一个 .py 文件可以作为模块被其他 Python 文件导入。

utils.py：

```python
def save_metrics(metrics):
    print(metrics)
```

train.py：

```python
from utils import save_metrics

save_metrics({"loss": 0.2})
```

import 让代码复用其他文件中的变量、函数或类。

### 8.2 常见 import 写法

```python
import numpy as np
from pathlib import Path
from package.module import function
```

import numpy as np 给模块起别名；from ... import ... 只导入指定对象。

不要随意使用 from module import *，因为它会让名字来源不清楚。

### 8.3 包与模块搜索

大型项目通常是：

```text
project/
├── train.py
├── configs/
├── src/
│   └── project/
│       ├── __init__.py
│       ├── data.py
│       └── model.py
└── requirements.txt
```

当 import 失败时，问题可能是：

- 当前工作目录不对。
- 包未安装。
- 包路径没有加入 Python 搜索路径。
- 模块名或文件名拼写错。
- 代码有循环 import。

环境和路径问题见 [[Conda与环境管理]]、[[Linux与Shell]]。

### 8.4 if __name__ == "__main__"

一个常见入口：

```python
def main():
    print("run experiment")

if __name__ == "__main__":
    main()
```

当直接执行 python train.py 时，__name__ 通常是 "__main__"，于是调用 main；当该文件被 import 时，main 不会自动执行。

这解释了为什么“文件被执行”不等于“其中所有函数都执行”。入口脚本经常使用这种结构。

## 9. 类、对象与继承

### 9.1 为什么需要 class？

函数适合封装一次操作；class 适合把数据和相关操作组织成一个对象。

```python
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, amount=1):
        self.value += amount

counter = Counter()
counter.add()
print(counter.value)
```

counter 是对象；value 是属性；add 是方法；__init__ 在对象创建时初始化状态。

### 9.2 self 是什么？

self 表示当前对象本身。调用 counter.add() 时，Python 会把 counter 作为 self 传给方法。

```python
class Sample:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value
```

不同对象可以拥有不同属性值，但共享同一个类定义的方法。

### 9.3 继承

继承表示一个类在另一个类的基础上扩展或修改行为。

```python
class Animal:
    def speak(self):
        return "unknown"

class Robot(Animal):
    def speak(self):
        return "beep"

robot = Robot()
print(robot.speak())
```

AI 代码中会看到类似关系：

- Dataset 子类实现数据读取接口。
- nn.Module 子类定义模型。
- Trainer 子类扩展训练流程。

PyTorch 的 nn.Module、Dataset 和 DataLoader 见 [[PyTorch]]；本篇只建立 Python class 的理解。

### 9.4 属性与方法的阅读方法

看到：

```python
model.encoder.forward(batch)
```

可以拆成：

- model 是对象。
- encoder 可能是 model 的属性。
- forward 是 encoder 对象的方法。
- batch 是传给方法的参数。

读科研代码时，沿着对象属性和方法定义跳转，比只看调用行更重要。

## 10. 异常、assert 与文件操作

### 10.1 异常是什么？

异常是运行时无法按当前路径继续执行时报告的问题。

```python
path = "missing.txt"
with open(path) as file:
    text = file.read()
```

如果文件不存在，可能产生 FileNotFoundError。不要把所有异常都吞掉；完整错误信息是 Debug 证据。

### 10.2 try / except

```python
try:
    value = int("abc")
except ValueError as error:
    print("invalid integer:", error)
```

只捕获你能合理处理的异常。过宽的 except Exception 可能让真正错误被隐藏。

### 10.3 raise

```python
def require_positive(x):
    if x <= 0:
        raise ValueError("x must be positive")
    return x
```

主动 raise 能把无效输入更早报告出来。

### 10.4 assert

```python
def process_batch(images):
    assert images.ndim == 4, images.shape
    return images
```

assert 适合检查程序设计中必须成立的条件。对用户输入和必须始终执行的业务验证，也可以使用明确的 if + raise。

### 10.5 with 与文件

```python
from pathlib import Path

path = Path("notes.txt")
path.write_text("experiment started", encoding="utf-8")
text = path.read_text(encoding="utf-8")
```

with 语句可以确保资源在离开代码块后被处理：

```python
with open("notes.txt", "r", encoding="utf-8") as file:
    text = file.read()
```

路径和当前目录细节见 [[Linux与Shell]]。

## 11. 列表推导与生成器

### 11.1 列表推导

```python
squares = [x * x for x in range(5)]
positive = [x for x in values if x > 0]
```

它适合简单的“遍历并构造列表”。逻辑太复杂时，普通 for 循环通常更容易 Debug。

### 11.2 生成器

生成器逐个产生值，不必一次把全部结果放入内存：

```python
def count_up_to(n):
    for i in range(n):
        yield i

for value in count_up_to(3):
    print(value)
```

大型数据集和数据加载器常使用“按需产生样本”的思想。PyTorch DataLoader 的工程行为见 [[PyTorch]]。

## 12. AI Research Code 的文件职责

看到一个项目时，可以先按职责分类：

| 文件/目录 | 可能职责 |
|---|---|
| train.py | 训练入口、参数解析和流程组织 |
| evaluate.py | 评估入口 |
| dataset.py/data.py | 数据读取、预处理、样本接口 |
| model.py | 模型类和 forward |
| config.yaml | 实验参数 |
| utils.py | 可复用工具函数 |
| scripts/ | Shell 启动脚本 |
| checkpoints/ | 模型状态文件 |
| outputs/ | 日志和结果 |

一个典型入口的伪代码：

```python
def main():
    config = load_config()
    dataset = build_dataset(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    train(model, dataset, optimizer, config)

if __name__ == "__main__":
    main()
```

读代码时先画出调用关系：

```text
main
├── load_config
├── build_dataset
├── build_model
├── build_optimizer
└── train
    ├── batch
    ├── model(batch)
    ├── loss
    └── optimizer.step()
```

Python 基础的最终目的，是让你能沿着这张图继续进入模型和数据的具体实现。

## 13. 与其他笔记的关系

本篇重点负责：

- Python 变量、类型和表达式
- list、tuple、dict、set
- 条件、循环、函数和作用域
- 模块、import、项目入口
- class、对象、方法和继承
- 异常、文件和工程代码结构

不在本篇系统展开：

- 解释器、环境和依赖 → [[Conda与环境管理]]
- 终端、路径、进程和 Shell → [[Linux与Shell]]
- Git 仓库、分支和 commit → [[Git]]
- traceback、日志和通用 Debug → [[Debug与工程基础]]
- ndarray、广播和数组 shape → [[NumPy]]
- Tensor、autograd、Dataset、DataLoader、nn.Module 和训练 → [[PyTorch]]

本篇会用 Dataset、Model、Tensor 等名字建立联系，但不提前讲完整深度学习框架。

## 14. 本篇知识地图

```text
Python 代码
├── 对象与类型
│   ├── 数字/字符串/布尔
│   └── list/tuple/dict/set
├── 控制流程
│   ├── if
│   ├── for/while
│   └── enumerate/zip
├── 函数
│   ├── 参数
│   ├── return
│   └── 作用域
├── 模块与工程
│   ├── import
│   ├── 包
│   └── main 入口
├── 面向对象
│   ├── class
│   ├── 属性/方法
│   └── 继承
├── 可靠运行
│   ├── 异常
│   ├── assert
│   └── 文件/路径
└── AI 项目
    ├── config
    ├── dataset
    ├── model
    ├── train
    └── evaluation
```

## 15. 快速查询表

| 我要做什么 | Python 写法 | 作用 |
|---|---|---|
| 查看类型 | type(x) | 确认对象类型 |
| 查看对象表示 | repr(x) | 显示更完整的值 |
| 查看长度 | len(x) | 查看容器或字符串长度 |
| 条件判断 | if condition: | 根据条件执行分支 |
| 遍历 | for item in items: | 逐个处理对象 |
| 获取索引和值 | enumerate(items) | 同时遍历位置和值 |
| 定义函数 | def name(...): | 封装可复用逻辑 |
| 返回结果 | return value | 从函数返回对象 |
| 导入模块 | import module | 使用其他文件/包 |
| 抛出错误 | raise ValueError(...) | 主动报告非法状态 |
| 检查条件 | assert condition | 尽早验证前置条件 |
| 文件路径 | Path("file") | 面向对象地处理路径 |
| 程序入口 | if __name__ == "__main__": | 直接执行时启动 main |

## 16. 练习题（题目下附参考答案）

### Level 1：概念

1. 变量、对象和类型分别是什么？

   **参考答案：**变量是指向对象的名字；对象是运行时实际存在的数据；类型描述对象能做什么以及如何解释其内容。

2. list、tuple、dict、set 的主要区别是什么？

   **参考答案：**list 有序且可修改；tuple 有序且通常不修改；dict 保存键值映射；set 保存不重复元素且不保证稳定顺序。

3. 为什么 Python 代码的缩进很重要？

   **参考答案：**缩进表示代码块层级，决定 if、for、函数和类中哪些语句属于同一块；缩进错误会改变逻辑或直接报错。

### Level 2：判断

4. 判断：Python 变量一旦第一次赋值，就不能再指向其他类型。
   
   **参考答案：**错误。Python 是动态类型语言，变量之后可以指向不同类型对象，但这种做法可能降低代码可读性。

5. 判断：函数没有显式 return 时，调用结果一定是空字符串。
   
   **参考答案：**错误。没有显式 return 的函数返回 None。

6. 判断：把列表赋值给另一个变量后，两个变量一定是互相独立的。
   
   **参考答案：**错误。普通赋值可能让两个变量指向同一个可变列表，修改一方会影响另一方。

7. 判断：import 一个文件时，该文件中的所有函数都会自动执行。
   
   **参考答案：**错误。函数定义通常只是创建函数；只有顶层执行语句会执行。使用 main 入口可以控制直接执行和被导入时的行为。

### Level 3：读代码

8. 下面代码最后打印什么？为什么？
   
   ```python
   a = [1, 2]
   b = a
   b.append(3)
   print(a)
   ```
   
   **参考答案：**打印 [1, 2, 3]。a 和 b 指向同一个列表对象，append 修改了该对象。

9. 下面代码中 config["seed"] 和 config.get("seed", 0) 有什么区别？
   
   ```python
   config = {"epochs": 10}
   ```
   
   **参考答案：**config["seed"] 在键不存在时触发 KeyError；get 在键不存在时返回默认值 0。是否使用默认值要根据配置设计决定，不能隐藏必填参数缺失。

10. 解释下面函数的输入和输出：
    
    ```python
    def normalize(x, maximum=255.0):
        return x / maximum
    ```
    
    **参考答案：**函数接收 x 和可选的 maximum，返回 x 除以 maximum 的结果；maximum 未传入时使用 255.0。

11. 下面入口代码的作用是什么？
    
    ```python
    def main():
        print("run")

    if __name__ == "__main__":
        main()
    ```
    
    **参考答案：**直接执行该文件时调用 main；被其他模块 import 时通常不自动调用 main，避免导入产生训练等副作用。

### Level 4：应用

12. 为什么 AI 项目要把读取数据、创建模型和训练循环拆成函数？
    
    **参考答案：**函数可以封装职责、减少重复、便于单独测试和替换，也能让入口脚本更容易阅读。

13. 设计一个 config 字典，至少包含 batch size、learning rate、device 和 seed。
    
   **参考答案：**例如 `{"batch_size": 32, "learning_rate": 0.0001, "device": "cuda", "seed": 42}`。实际项目还应确认每个字段的类型和允许范围。

14. 如何从一个训练项目入口开始阅读代码？
    
    **参考答案：**先找到 main 或命令行入口，记录读取的配置；再追踪 build_dataset、build_model、build_optimizer 和 train/evaluate，分别查看它们的参数、返回值和调用关系。

### Level 5：Debug

15. 代码报 AttributeError: NoneType has no attribute to。你会怎么排查？
    
    **参考答案：**说明某个本应是 Tensor/对象的变量实际是 None。向前追踪它由哪个函数返回，检查是否忘记 return、某个分支没有返回值，或配置导致初始化失败。

16. 代码报 KeyError: learning_rate。第一轮应该检查什么？
    
    **参考答案：**打印配置对象和 keys，确认配置文件是否读取成功、键名拼写是否正确、默认配置是否覆盖，以及当前代码版本是否匹配该配置。

17. 函数调用后结果是 None，但你以为它返回了数据。如何定位？
    
    **参考答案：**检查函数所有分支是否都有 return；确认返回值没有只在条件内部产生；在调用处打印 type 和 repr，并检查是否误用了只打印不返回的函数。

### Level 6：综合科研场景

18. 一个 World Model 的 train.py 看起来很长。你如何建立阅读顺序？
    
    **参考答案：**先找入口和 main；再识别 config、dataset、model、optimizer、train loop 和 checkpoint；画函数调用图；逐个进入定义，记录输入输出和关键状态，不要从第一行盲目读到最后一行。

19. 你要写一个函数读取一个样本并检查它。至少检查哪些内容？
    
    **参考答案：**检查样本是否存在、类型、字段键、数组/Tensor shape、dtype、数值范围、NaN/Inf 和标签合法性；必要时用 assert 或明确异常报告。

20. 设计一次“从 Python 脚本到训练入口”的最小验证。
    
    **参考答案：**确认环境和工作目录；导入关键模块；读取最小配置；构造一个样本/小数据集；创建模型对象；执行一次 forward 或训练函数；打印输入输出类型/shape 和结果，并记录完整命令。

## 17. 掌握度检查（附答案）

1. Python 中变量和对象有什么区别？
   
   **答案：**变量是名字，对象是实际数据；变量可以重新指向不同对象。

2. 为什么要理解类型？
   
   **答案：**类型决定对象的表示、可用操作和函数接口；许多错误首先是类型不符合预期。

3. list 和 NumPy ndarray 有什么边界？
   
   **答案：**list 是通用 Python 容器；ndarray 是规则数值数组，适合向量化和科学计算。数组数据基础见 [[NumPy]]。

4. 字典为什么常用于 AI 配置？
   
   **答案：**字典可以用清晰的键名映射 batch size、学习率、设备等参数，便于读取和传递。

5. if、for 和函数分别解决什么问题？
   
   **答案：**if 做条件分支；for 重复处理序列；函数封装可复用的输入到输出逻辑。

6. return 和 print 有什么区别？
   
   **答案：**print 把信息显示出来；return 把对象交给调用者。训练代码需要 return 才能继续使用结果。

7. 为什么 import 和直接运行 Python 文件的行为可能不同？
   
   **答案：**import 会加载模块但通常不执行 main 入口；直接运行时 __name__ 为 "__main__"，可以触发入口代码。

8. class、对象、属性、方法如何对应？
   
   **答案：**class 是对象的定义模板；对象是实例；属性保存对象状态；方法是对象可以调用的行为。

9. 继承在 AI 代码中可能出现在哪里？
   
   **答案：**Dataset 子类、nn.Module 模型子类、训练器和数据处理器都可能通过继承复用或扩展基础类。

10. 为什么不应该随意捕获所有异常？
    
    **答案：**过宽的异常捕获可能隐藏真正根因，让程序继续在错误状态中运行；应只处理明确知道如何处理的异常。

11. assert 在工程代码中有什么价值？
    
    **答案：**它能在错误数据继续传播前检查 shape、类型、范围和前置条件，让错误更接近根因暴露。

12. 为什么研究代码要尽量少依赖全局变量？
    
    **答案：**全局状态难以追踪、测试和复现；显式传递配置、数据和模型更容易理解和 Debug。

13. 如何阅读一个不熟悉的 Python AI 仓库？
    
    **答案：**先看 README 和入口脚本，再追踪配置、数据、模型、训练和评估的调用关系；用 type、shape 和返回值记录关键接口。

14. Python 基础如何连接到 PyTorch？
    
    **答案：**Python 的 class 可用于定义 nn.Module，函数可封装训练步骤，list/dict 可组织配置和样本，模块系统组织 Dataset、Model 和 train 文件。

15. 怎样判断自己已经具备用 Python 阅读科研代码的基础？
    
    **答案：**能够读懂入口、追踪函数调用、解释变量和数据结构、判断对象类型、理解 class 方法和异常，并能用最小输入验证代码行为。
