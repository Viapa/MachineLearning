"""
目标: 实现一个管理员类, 用于注册和登录学生管理系统
需求: 至少包含管理员账号id, 姓名userName, 密码password
"""

class Administor(object):
    """使用封装模式进行管理员类的定义"""
    def __init__(self, id, userName, password):
        self.__id = id;
        self.__userName = userName;
        self.__password = password;

    def getId(self):
        return self.__id;

    def getUserName(self):
        return self.__userName;

    def setUserName(self, userName):
        self.__userName = userName;

    def getPassword(self):
        return self.__password;

    def setPassword(self, password):
        self.__password = password;


### 测试 ###
if __name__ == "__main__":
    test = Administor("12345", "华强", "541283");
    test.setUserName("疯子")
    name = test.getUserName()
    print(name)