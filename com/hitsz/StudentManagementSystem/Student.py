"""
目标: 实现一个学生类, 用于存储学生信息
需求: 至少包含学号id, 姓名name, 性别sex, 年龄age, 身高height, 年级grade, 绩点gpa
"""


class Student(object):
    """使用封装模式进行学生类的定义"""
    def __init__(self, id, name, sex, age, height, grade, gpa):
        self.__id = id;
        self.__name = name;
        self.__sex = sex;
        self.__age = age;
        self.__height = height;
        self.__grade = grade;
        self.__gpa = gpa;

    def getId(self):
        return self.__id;

    def setId(self, id):
        self.__id = id;

    def getName(self):
        return self.__name;

    def setName(self, name):
        self.__name = name;

    def getSex(self):
        return self.__sex;

    def setSex(self, sex):
        self.__sex = sex;

    def getAge(self):
        return self.__age;

    def setAge(self, age):
        self.__age = age;

    def getHeight(self):
        return self.__height;

    def setHeight(self, height):
        self.__height = height;

    def getGrade(self):
        return self.__grade;

    def setGrade(self, grade):
        self.__grade = grade;

    def getGpa(self):
        return self.__gpa;

    def setGpa(self, gpa):
        self.__gpa = gpa;


### 测试 ###
if __name__ == "__main__":
    test = Student("19S055011", "张珂", "男", 25, 183, "硕士二年级", 3.32);
    test.setAge(24)
    age = test.getAge()
    print(age)