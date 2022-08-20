
"""
目标：学会赋值, 浅拷贝和深拷贝的区别和用法
"""

### 赋值操作 ###
## 作用：令不同变量指向同一对象; 对象改变,变量值也随之改变

class Computer(object):
    def __init__(self, disk, cpu):
        self.disk = disk;
        self.cpu = cpu;

c1 = Computer(8, "I7-9700");
c2 = c1;
print(id(c1), id(c2)) # 二者完全一样, 因为赋值的是类的地址

c2.cpu = "I9-10100K"; # 既然是地址, 指向的内容相同, 那么改变后其他指向该地址的变量值随之也改变
print(c1.cpu)
print("-----------------------------")

### 浅拷贝操作 ###
## 作用：原对象和拷贝对象都会有相同的子对象

class Disk():
    pass;

class Cpu():
    pass;

d1 = Disk();
c1 = Cpu();
t1 = Computer(d1, c1);

# 浅拷贝
import copy
t2 = copy.copy(t1);
print(id(t1), t1.cpu, t1.disk) # 对实例化对象id不同, 但是子类id一致
print(id(t2), t2.cpu, t2.disk)
