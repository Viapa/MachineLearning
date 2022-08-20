"""
目标：学习OS模块的常见使用操作, 这是一个与系统操作相关的模块
"""

import os

### 系统执行命令 ###

# os.system("notepad.exe")  打开应用操作, 相当于调取windows的运行模块
# os.system("calc.exe") 计算器
# os.startfile("D:\腾讯QQ\Bin\qq.exe") 打开qq软件(一切可执行文件)

### 创建文件, 设置路径操作 ###

print(os.getcwd()); # 返回当前的工作绝对路径
print(os.listdir("D:\python")) # 返回指定路径下的文件和目录信息
os.mkdir("new_dir") # 在指定目录下创建一个文件夹
os.makedirs("A/B/C") # 在指定目录下创建多级文件夹(a->b->c)
os.rmdir("new_dir") # 删除指定目录下的一个文件夹
os.removedirs("A/B/C") # 删除指定路径下的多级文件夹
# os.chdir(path) 设定指定路径path为当前的工作目录
print("-----------------------")

### os.path 模块操作 ###

print(os.path.abspath("OS.py")) # 获取文件的绝对路径
print(os.path.exists("OS1.py")) # 判断路径下的文件是否存在, 返回boolean类型
print(os.path.join("D:\python", "test.py")) # 将前序路径和后序路径进行拼接操作
print(os.path.split("D:\python\code\com\hitsz\object\OS.py")) # 将文件和前序路径目录分离
print(os.path.splitext("OS.py")) # 将文件名和文件类型后缀进行分离
print(os.path.basename("D:\python\code\com\hitsz\object\OS.py")) # 从一个路径中提取出文件名称
print(os.path.dirname("D:\python\code\com\hitsz\object\OS.py")) # 从一个路径中提取出文件的前序目录, 不包括文件
print(os.path.isdir("D:\python\code\com\hitsz\object\OS.py")) # 判断路径是否为目录(文件的前序目录)
print(os.path.isdir("D:\python\code\com\hitsz\object"))
print(os.path.isfile("D:\python\code\com\hitsz\object\OS.py")) # 用于判断对象是否为一个文件。
