"""
目标: 使用os模块输出指定目录下的所有py文件
"""

import os
path = os.getcwd();
lst = os.listdir(path);
for (file) in lst:
    if file.endswith(".py"):
        print(file)

print("---------------------------")


"""
目标: 使用os.walk()方法依次遍历目录中的所有文件，方向为根目录->子目录->....
"""

path = "D:\python\code\com\hitsz";
for root, dirs, files in os.walk(path, topdown=True):
    print(root)
    for dir in dirs:
        print(os.path.join(root, dir))
    for file in files:
        print(os.path.join(root, file))
    print("-----------遍历迭代(进入下一层目录)------------")