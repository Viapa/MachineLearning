from Administrator import Administor
from Student import Student
import numpy as np
import random
import pickle

class StudentManagementSystem(object):
    """
    目标: 主界面显示模块, 能够基于用户的输入进入指定页面
    功能:
    <学生信息管理界面>
    0 = 退出系统
    1 = 添加学生信息
    2 = 查询学生信息
    3 = 删除学生信息
    4 = 修改学生信息
    5 = 基于数据排序
    6 = 学生信息统计
    7 = 学生信息存储
    <登录系统界面>
    1 = 登陆账号
    2 = 注册账号
    3 = 信息存储
    4 = 退出系统
    """
    # 管理员主页面
    def adminMain(self, adminList, studentList):
        while True:
            print("===============欢迎进入HIT学生管理系统===============");
            print("请您先登录管理员系统: ")
            print("1 -> 登录账号: ")
            print("2 -> 注册账号: ")
            print("3 -> 管理员账号存储: ")
            print("4 -> 退出系统: ")
            command = str(input())
            if (command == '1'):
                self.login(adminList, studentList);
            elif (command == '2'):
                self.register(adminList);
            elif (command == '3'):
                self.storage(adminList);
            elif (command == '4'):
                return;
            else:
                print("您的输入有误, 请重新输入!")

    # 管理员账户登录
    def login(self, adminList, studentList):
        if (len(adminList) == 0):
            print("提示: ")
            print("该系统还没有创建任何管理员, 请您先注册后再登录!")
            return;

        print("===============欢迎进入用户登录界面===============")
        while True:
            print("请输入您的用户id: ")
            userId = str(input())
            account = self.getAccountById(adminList, userId)
            if (account != None):
                print("请输入您的密码: ")
                password = str(input())
                if (password == account.getPassword()):
                    print("登录成功! 欢迎您, {:} 先生/女士！".format(account.getUserName()))
                    self.showAdminCommand(studentList);
                    return;
                else:
                    print("对不起, 您输入的密码有误, 请重新输入!")
            else:
                print("对不起, 系统中无此管理账号!")

    # 管理员账户注册
    def register(self, adminList):
        print("===============欢迎进入用户注册界面===============")
        print("请输入您的用户名: ")
        userName = str(input())
        while True:
            print("请输入您的登录密码: ")
            password = str(input())
            print("请再一次确认您的密码: ")
            okPassword = str(input())
            if (password == okPassword):
                break;
            else:
                print("您两次输入的密码不一致, 请重新输入!")

        userId = self.creatUserId(adminList)
        account = Administor(userId, userName, okPassword)
        adminList.append(account);
        print("恭喜您, 用户名为: {:} 的管理员已创建成功! 您的id账号是: {:}, 欢迎您登录学生管理系统!".format(userName, userId))

    # 管理员信息存储到本地
    def storage(self, adminList):
        dir_path = "D:\python\data\STUSYS"
        file_name = "\Administor.pickle";
        file_path = dir_path + file_name;
        with open(file = file_path, mode = "wb") as f:
            for admin in adminList:
                obj = pickle.dumps(admin)
                f.write(obj)
        print("管理员信息已成功存储到路径 {:} 中!".format(file_path))

    def creatUserId(self, adminList):
        # 随机生成8位不重复的账号
        while True:
            userId = "";
            for i in range(8):
                userId += str(random.randint(0, 9));
            account = self.getAccountById(adminList, userId);
            if not account:
                return userId;

    def getAccountById(self, adminList, userId):
        for i in range(len(adminList)):
            acc = adminList[i]
            if (acc.getId() == userId):
                return acc;
        return None;

    def showAdminCommand(self, studentList):
        while True:
            print("===============学生管理系统功能菜单===============");
            print("1、添加学生信息")
            print("2、查询学生信息")
            print("3、删除学生信息")
            print("4、修改学生信息")
            print("5、学生数据排序")
            print("6、学生信息统计")
            print("7、学生信息存储")
            print("0、退出管理员界面")
            command = str(input())
            if (command == '1'):
                self.addInfo(studentList);
            elif (command == '2'):
                self.queryInfo(studentList);
            elif (command == '3'):
                self.removeInfo(studentList);
            elif (command == '4'):
                self.adjustInfo(studentList);
            elif (command == '5'):
                self.sortInfo(studentList);
            elif (command == '6'):
                self.staticInfo(studentList);
            elif (command == '7'):
                self.storeInfo(studentList);
            elif (command == '0'):
                return; # 退出管理员操作
            else:
                print("您的输入有误, 请重新输入!")

    def addInfo(self, studentList):
        print("===============学生信息添加界面================")
        while True:
            print("请输入学生学号: ")
            stuId = str(input())
            print("请输入学生姓名: ")
            stuName = str(input())
            print("请输入学生性别 (男/女): ")
            stuSex = str(input())
            print("请输入学生年龄: ")
            stuAge = str(input())
            print("请输入学生身高 (cm, 可选): ")
            stuHeight = str(input())
            print("请输入学生年级 (可选): ")
            stuGrade = str(input())
            print("请输入学生学业绩点: ")
            stuGpa = str(input())

            print("请确认以下学生信息是否正确, 按1确认添加, 按0取消添加: ")
            print("学号\t\t\t姓名\t\t\t性别\t\t\t年龄\t\t\t身高\t\t\t年级\t\t\t绩点")
            print("{:}\t\t\t{:}\t\t\t{:}\t\t\t{:}\t\t\t{:}\t\t\t{:}\t\t\t{:}".format(stuId, stuName, stuSex, stuAge, stuHeight, stuGrade, stuGpa))
            command = str(input())
            if (command == '1'):
                isExist = self.getStudentById(studentList, stuId) # 判断该学生是否已在系统中
                if (isExist):
                    print("该学生已在系统中, 请勿重复添加!")
                    break;
                else:
                    student = Student(stuId, stuName, stuSex, stuAge, stuHeight, stuGrade, stuGpa)
                    studentList.append(student);
                    print("学生 {:} 的信息添加成功!".format(student.getName()))
                    while True:
                        print("您是否还要继续添加？ (按1继续, 按0退出)")
                        flag = str(input())
                        if (flag == '1'):
                            break;
                        elif (flag == '0'):
                            return;
                        else:
                            print("您的输入有误, 请重新输入!")

            elif (command == '0'):
                while True:
                    print("您是否还要继续添加？ (按1继续, 按0退出)")
                    flag = str(input())
                    if (flag == '1'):
                        break;
                    elif (flag == '0'):
                        return;
                    else:
                        print("您的输入有误, 请重新输入!")
            else:
                print("您的输入有误, 请重新输入!")

    def getStudentById(self, studentList, stuId):
        for i in range(len(studentList)):
            stu = studentList[i]
            if (stu.getId() == stuId):
                return stu;
        return None;

    def getStudentByName(self, studentList, stuName):
        for i in range(len(studentList)):
            stu = studentList[i]
            if (stu.getName() == stuName):
                return stu;
        return None;

    def queryInfo(self, studentList):
        print("===============学生信息查询界面================")
        while True:
            print("请输入您想要查询的学生学号或姓名: ")
            idOrName = str(input())
            id_out = self.getStudentById(studentList, idOrName);
            name_out = self.getStudentByName(studentList, idOrName);
            if (id_out != None and name_out == None):
               stu = id_out;
            elif (id_out == None and name_out != None):
                stu = name_out;
            else:
                stu = None;
                print("对不起, 您所查询的学生不在信息系统中!")
                while True:
                    print("您是否还要继续查询? (Y/N)")
                    command = str(input())
                    if (command == 'y' or command == 'Y'):
                        break;
                    elif (command == 'n' or command == 'N'):
                        return;
                    else:
                        print("您的输入有误, 请重新输入!")
            if stu:
                print("查询成功, 系统中的该学生信息如下: ")
                self.showStudentInfo(stu)
                while True:
                    print("您是否还要继续查询? (Y/N)")
                    command = str(input())
                    if (command == 'y' or command == 'Y'):
                        break;
                    elif (command == 'n' or command == 'N'):
                        return;
                    else:
                        print("您的输入有误, 请重新输入!")

    def removeInfo(self, studentList):
        print("===============学生信息移除界面================")
        isDone = False;
        while True:
            print("请输入您想要删除的学生学号或姓名: ")
            idOrName = str(input())
            id_out = self.getStudentById(studentList, idOrName);
            name_out = self.getStudentByName(studentList, idOrName);
            if (id_out != None and name_out == None):
                stu = id_out;
            elif (id_out == None and name_out != None):
                stu = name_out;
            else:
                stu = None;
                print("对不起, 您要删除的学生不在信息系统中!")
                while True:
                    print("您是否还要继续查询并删除? (Y/N)")
                    command = str(input())
                    if (command == 'y' or command == 'Y'):
                        break;
                    elif (command == 'n' or command == 'N'):
                        return isDone;
                    else:
                        print("您的输入有误, 请重新输入!")
            if stu:
                print("系统中的该学生信息如下: ")
                self.showStudentInfo(stu);
                print("您是否确认要删除 (Y/N)?")
                flag = str(input())
                if (flag == "y" or flag == "Y"):
                    studentList.remove(stu);
                    print("删除成功!")
                    isDone = True;
                    while True:
                        print("您是否还要继续删除? (Y/N)")
                        command = str(input())
                        if (command == 'y' or command == 'Y'):
                            break;
                        elif (command == 'n' or command == 'N'):
                            return isDone;
                        else:
                            print("您的输入有误, 请重新输入!")
                elif (flag == "n" or flag == "N"):
                    print("您已取消操作...")
                    while True:
                        print("您是否还要继续删除? (Y/N)")
                        command = str(input())
                        if (command == 'y' or command == 'Y'):
                            break;
                        elif (command == 'n' or command == 'N'):
                            return isDone;
                        else:
                            print("您的输入有误, 请重新输入!")
                else:
                    print("您的输入有误, 请重新输入!")


    def adjustInfo(self, studentList):
        print("===============学生信息修改界面================")
        while True:
            print("请输入您想要修改的学生学号: ")
            stuId = str(input())
            stu = self.getStudentById(studentList, stuId);
            if (stu):
                print("您当前正在修改学生 {:} 的个人信息.".format(stu.getName()))
                print("<-请选择您要修改的信息项-> ")
                print("1. 学号, 2. 姓名, 3. 性别, 4. 年龄, 5. 身高, 6. 年级, 7. 绩点")
                choice = str(input())
                if (choice == "1"):
                    print("该学生的旧学号为: {:}".format(stu.getId()));
                    print("请您输入新的学号: ")
                    stu.setId(str(input()))
                elif (choice == "2"):
                    print("该学生的旧姓名为: {:}".format(stu.getName()));
                    print("请您输入新的姓名: ")
                    stu.setName(str(input()))
                elif (choice == "3"):
                    print("该学生的旧性别为: {:}".format(stu.getSex()));
                    print("请您输入新的性别: ")
                    stu.setSex(str(input()))
                elif (choice == "4"):
                    print("该学生的旧年龄为: {:}".format(stu.getAge()));
                    print("请您输入新的年龄: ")
                    stu.setAge(str(input()))
                elif (choice == "5"):
                    print("该学生的旧身高为: {:}".format(stu.getHeight()));
                    print("请您输入新的身高: ")
                    stu.setHeight(str(input()))
                elif (choice == "6"):
                    print("该学生的旧年级为: {:}".format(stu.getGrade()));
                    print("请您输入新的年级: ")
                    stu.setGrade(str(input()))
                elif (choice == "7"):
                    print("该学生的旧绩点为: {:}".format(stu.getGpa()));
                    print("请您输入新的绩点: ")
                    stu.setGpa(str(input()))
                else:
                    print("您的输入有误, 请重新输入!")
                    continue;

                print("修改成功!该学生信息更新如下: ")
                self.showStudentInfo(stu)
                while True:
                    print("您是否还要继续修改? (Y/N)")
                    command = str(input())
                    if (command == 'y' or command == 'Y'):
                        break;
                    elif (command == 'n' or command == 'N'):
                        return;
                    else:
                        print("您的输入有误, 请重新输入!")

            else:
                print("对不起, 该学号不在信息系统中!")
                while True:
                    print("您是否还要继续修改? (Y/N)")
                    command = str(input())
                    if (command == 'y' or command == 'Y'):
                        break;
                    elif (command == 'n' or command == 'N'):
                        return;
                    else:
                        print("您的输入有误, 请重新输入!")

    def showStudentInfo(self, student):
        print("学号: {:}".format(student.getId()))
        print("姓名: {:}".format(student.getName()))
        print("性别: {:}".format(student.getSex()))
        print("年龄: {:}".format(student.getAge()))
        print("身高: {:}".format(student.getHeight()))
        print("年级: {:}".format(student.getGrade()))
        print("绩点: {:}".format(student.getGpa()))

    def sortInfo(self, studentList):
        print("===============学生数据排序界面================")
        while True:
            print("您想要基于以下哪项数据进行排序?")
            print("1. 学号, 2. 性别, 3. 年龄, 4. 身高, 5. 绩点")
            choice = str(input())
            if choice not in ["1", "2", "3", "4", "5"]:
                print("您的输入有误, 请重新输入!")
                continue;

            print("您想要选择哪种排序方式? (按1升序, 按0降序)")
            reverse = str(input())
            if (reverse == "1"):
                reverse = False;
            elif (reverse == "0"):
                reverse = True;
            else:
                print("您的输入有误, 请重新输入!")
                continue;

            if (choice == "1"):
                studentList = sorted(studentList, key=lambda x: x.getId(), reverse=reverse)
            elif (choice == "2"):
                studentList = sorted(studentList, key=lambda x: x.getSex(), reverse=reverse)
            elif (choice == "3"):
                studentList = sorted(studentList, key=lambda x: x.getAge(), reverse=reverse)
            elif (choice == "4"):
                studentList = sorted(studentList, key=lambda x: x.getHeight(), reverse=reverse)
            elif (choice == "5"):
                studentList = sorted(studentList, key=lambda x: x.getGpa(), reverse=reverse)

            print("您选择的排序结果如下: ")
            print("学号\t\t\t姓名\t\t\t性别\t\t\t年龄\t\t\t身高\t\t\t年级\t\t\t绩点")
            for s in studentList:
                print("{:}\t\t\t{:}\t\t\t{:}\t\t\t{:}\t\t\t{:}\t\t\t{:}\t\t\t{:}".format(s.getId(), s.getName(), s.getSex(), s.getAge(),
                                                                                               s.getHeight(), s.getGrade(), s.getGpa()))
            while True:
                print("您是否还要继续排序? (Y/N)")
                command = str(input())
                if (command == 'y' or command == 'Y'):
                    break;
                elif (command == 'n' or command == 'N'):
                    return;
                else:
                    print("您的输入有误, 请重新输入!")

    def staticInfo(self, studentList):
        print("===============学生数据统计界面================")
        while True:
            print("您想要基于以下哪项数据进行统计分析?")
            print("1. 性别, 2. 年龄, 3. 身高, 4. 绩点")
            choice = str(input())
            if choice not in ["1", "2", "3", "4"]:
                print("您的输入有误, 请重新输入!")
                continue;

            if (choice == "1"):
                print("您想要选择哪种统计方式?")
                print("1. 男、女学生人数占比")
                print("2. 男、女学生人员数量")
                c = str(input())
                numOfMan = 0;
                numOfWomen = 0;
                total = len(studentList)
                for s in studentList:
                    if (s.getSex() == "男"):
                        numOfMan += 1
                    elif (s.getSex() == "女"):
                        numOfWomen += 1
                if (c == "1"):
                    print("所有学生中, 男性占比{:.2f}%, 女性占比{:.2f}%".format(numOfMan / total * 100, numOfWomen / total * 100))
                elif (c == "2"):
                    print("所有学生中, 男生有 {:d} 个, 女生有 {:d} 个".format(numOfMan, numOfWomen))
                else:
                    print("您的输入有误, 请重新输入!")

            elif (choice == "2"):
                print("您想要选择哪种统计方式?")
                print("1. 学生的年龄均值")
                print("2. 学生的年龄中位数")
                print("3. 学生的年龄最值")
                print("4. 学生的年龄标准差")
                c = str(input())
                ages = [s.getAge() for s in studentList]
                ages = list(map(int, ages))
                if (c == "1"):
                    print("所有学生的年龄均值为: {:.4f}".format(np.mean(ages)))
                elif (c == "2"):
                    print("所有学生的年龄中位数为: {:d}".format(int(np.median(ages))))
                elif (c == "3"):
                    print("所有学生的年龄最大值为: {:d}, 最小值为: {:d}".format(int(np.max(ages)), int(np.min(ages))))
                elif (c == "4"):
                    print("所有学生的年龄标准差为: {:.4f}".format(np.std(ages)))
                else:
                    print("您的输入有误, 请重新输入!")

            elif (choice == "3"):
                print("您想要选择哪种统计方式?")
                print("1. 学生的身高均值")
                print("2. 学生的身高中位数")
                print("3. 学生的身高最值")
                print("4. 学生的身高标准差")
                c = str(input())
                heights = [s.getHeight() for s in studentList]
                heights = list(map(float, heights))
                if (c == "1"):
                    print("所有学生的身高均值为: {:.4f}".format(np.mean(heights)))
                elif (c == "2"):
                    print("所有学生的身高中位数为: {:d}".format(int(np.median(heights))))
                elif (c == "3"):
                    print("所有学生的身高最大值为: {:d}, 最小值为: {:d}".format(int(np.max(heights)), int(np.min(heights))))
                elif (c == "4"):
                    print("所有学生的身高标准差为: {:.4f}".format(np.std(heights)))
                else:
                    print("您的输入有误, 请重新输入!")

            elif (choice == "4"):
                print("您想要选择哪种统计方式?")
                print("1. 学生的绩点均值")
                print("2. 学生的绩点中位数")
                print("3. 学生的绩点最值")
                print("4. 学生的绩点标准差")
                c = str(input())
                gpas = [s.getGpa() for s in studentList]
                gpas = list(map(float, gpas))
                if (c == "1"):
                    print("所有学生的身高均值为: {:.4f}".format(np.mean(gpas)))
                elif (c == "2"):
                    print("所有学生的身高中位数为: {:d}".format(int(np.median(gpas))))
                elif (c == "3"):
                    print("所有学生的身高最大值为: {:d}, 最小值为: {:d}".format(int(np.max(gpas)), int(np.min(gpas))))
                elif (c == "4"):
                    print("所有学生的身高标准差为: {:.4f}".format(np.std(gpas)))
                else:
                    print("您的输入有误, 请重新输入!")

            while True:
                print("您是否还要继续进行统计? (Y/N)")
                command = str(input())
                if (command == 'y' or command == 'Y'):
                    break;
                elif (command == 'n' or command == 'N'):
                    return;
                else:
                    print("您的输入有误, 请重新输入!")

    def storeInfo(self, studentList):
        dir_path = "D:\python\data\STUSYS"
        file_name = "\Student.pickle";
        file_path = dir_path + file_name;
        with open(file = file_path, mode = "wb") as f:
            for stu in studentList:
                obj = pickle.dumps(stu)
                f.write(obj)
        print("学生信息已成功存储到路径 {:} 中!".format(file_path))



def loadInfo(file_path):
    arr = list();
    with open(file = file_path, mode = "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
                arr.append(obj)
            except:
                break;
    if arr:
        print("后台文件已读取成功!")
        return arr;
    else:
        print("数据读取错误, 请检查数据格式!")
        return []

if __name__ == "__main__":
    while True:
        print("是否需要加载后台已存信息? (Y/N)")
        command = str(input())
        if (command == "y" or command == "Y"):
            adminList = loadInfo("D:\python\data\STUSYS\Administor.pickle")
            studentList = loadInfo("D:\python\data\STUSYS\Student.pickle")
            StuMgSys = StudentManagementSystem();
            StuMgSys.adminMain(adminList, studentList)
            print("感谢您使用HIT学生信息管理系统, 祝您生活愉快。")
            break;
        elif (command == "n" or "N"):
            adminList = [];  # 管理员列表
            studentList = []; # 学生信息列表
            StuMgSys = StudentManagementSystem();
            StuMgSys.adminMain(adminList, studentList)
            print("感谢您使用HIT学生信息管理系统, 祝您生活愉快。")
            break;
        else:
            print("您的输入有误, 请重新输入!")
