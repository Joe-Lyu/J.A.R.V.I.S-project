"""
for windows!!!
initialize your environment automatically!
please make sure you are running the file in a virtual-env in order not to mess up your base environment
for main version v1.2.1
"""
import os


def install_package_check(package_name_list) -> (bool, list):
    flag = 1
    FalseList = []
    p = os.popen("pip list --format=columns")  # 获取所有包名 直接用 pip list 也可获取
    pip_list = p.read()  # 读取所有内容
    # print(pip_list)
    for packageName in package_name_list:
        package_name = packageName.replace("_", "-")  # 下载pip fake_useragent 包时  包名是:fake-useragent
        if package_name in pip_list:
            print("already installed {}".format(package_name))
        else:
            print("{} not installed!starting installation...".format(package_name))
            p = os.popen("pip install {}".format(package_name))
            if "Success" in p.read():
                # print(p.read())
                print("{} installed successfully!".format(package_name))
            else:
                print(p.read())
                flag = 0
                FalseList.append(package_name)
    return flag, FalseList

# print(install_package_check(['fake_useragent','ffahwf','dududud']))
# from fake_useragent import UserAgent
