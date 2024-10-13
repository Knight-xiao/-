# 空间机器人项目工作日志

## 2024.10.13工作总结:
本周主要任务是基于https://github.com/Tsinghua-Space-Robot-Learning-Group/SpaceRobotEnv，搭建空间机器人的仿真环境。

### 项目安装
下载上述链接中的项目，先安装mujoco，再按照readme安装，然后运行`test_env.py`检查是否安装成功，会提示`ModuleNotFoundError: No module named 'gym.envs.robotics'`，解决办法为安装老版本的gym(0.19.0),安装完后再运行，按照提示安装所需要的包即可。

### 项目管理
目前只需要搭建空间机器人仿真环境，因此主要关注`SpaceRobotEnv`文件夹，该文件夹下的`asset`文件夹是机器人模型文件，`envs`则构建了数个空间机器人仿真环境，具体详看链接的项目介绍。要添加我们自己的空间机器人环境，主要在这两个文件夹下修改。

### 具体修改
首先获取空间机器人模型，将其从urdf格式改为xml格式，方法可参照https://blog.csdn.net/Time_Memory_cici/article/details/138198171

