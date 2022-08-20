"""
目标: 学会理解和使用浅拷贝和深拷贝
"""

import copy

class People:
    def __init__(self, name):
        self.name = name;

class Name:
    pass


name = Name();

### 赋值操作 ### 作用：地址对象完全一样, 名称不同
P1 = People(name)
P2 = P1;
print("===============赋值操作结果=====================")
print("P1对象的地址为{:}, P2对象的地址为{:}.".format(id(P1), id(P2)));
print("P1对象Name子类的地址为{:}, P2对象Name子类的地址为{:}.".format(id(P1.name), id(P2.name)))

### 浅拷贝操作 ### 作用：类对象不同(新建), 但子类地址相同
print("===============浅拷贝操作结果=====================")
P1 = People(name)
P2 = copy.copy(P1);
print("P1对象的地址为{:}, P2对象的地址为{:}.".format(id(P1), id(P2)));
print("P1对象Name子类的地址为{:}, P2对象Name子类的地址为{:}.".format(id(P1.name), id(P2.name)))

### 深拷贝操作 ### 作用：类对象不同(新建), 并且其子类地址也不同(对全部子类都进行新建)
print("===============深拷贝操作结果=====================")
P1 = People(name)
P2 = copy.deepcopy(P1);
print("P1对象的地址为{:}, P2对象的地址为{:}.".format(id(P1), id(P2)));
print("P1对象Name子类的地址为{:}, P2对象Name子类的地址为{:}.".format(id(P1.name), id(P2.name)))