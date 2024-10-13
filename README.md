# 空间机器人项目工作日志

## 2024.10.13工作总结:
本周主要任务是基于https://github.com/Tsinghua-Space-Robot-Learning-Group/SpaceRobotEnv，搭建空间机器人的仿真环境。

### 项目安装
下载上述链接中的项目，先安装mujoco，再按照readme安装，然后运行`test_env.py`检查是否安装成功，会提示`ModuleNotFoundError: No module named 'gym.envs.robotics'`，解决办法为安装老版本的gym(0.19.0),安装完后再运行，按照提示安装所需要的包即可。

### 项目管理
目前只需要搭建空间机器人仿真环境，因此主要关注`SpaceRobotEnv`文件夹，该文件夹下的`asset`文件夹是机器人模型文件，`envs`则构建了数个空间机器人仿真环境，具体详看链接的项目介绍。要添加我们自己的空间机器人环境，主要在这两个文件夹下修改。

### 具体修改
#### asset修改
首先获取空间机器人模型，将其从urdf格式改为xml格式，方法可参照https://blog.csdn.net/Time_Memory_cici/article/details/138198171。
导出来的模型主要包含`asset`和`worldbody`属性，为了保证后续仿真，需要为每个关节添加`actuator`属性，格式大致如下：
```
<actuator>
        <general biasprm="0 0 -100" biastype="affine" ctrllimited="true" ctrlrange="-2.0942 2.0942" forcelimited="true" forcerange="-150 150" gainprm="100 0 0" joint="joint1" name="joint1_T" />
</actuator>
```
PS:若需要修改颜色，只需修改rgb值
然后添加目标卫星模块：现在`asset`里定义，再在`worldbody`里调用即可
再添加地球背景：找一张太空拍摄的地球圆形图片，**注意必须是png格式**，在`asset`里定义，再在`worldbody`里调用，大致如下：
```
    <asset>
        <!-- 地球背景设置 -->
        <texture name="earth" type="2d" file="stl/earth.png" />
        <material name="earth_material" texture="earth" texrepeat="1 1" rgba="1 1 1 1"/>
    </asset>
    <worldbody>
        <!-- 地球背景设置 -->
        <geom name="earth_background" type="sphere" size="50" material="earth_material" pos="-50 0 -30" euler="0 1.57 0" />
    <worldbody />
```
#### envs修改
参考`envs`下的`SpaceRobotDualArm.py`进行修改，复制该文件并改名为`SpaceRobotDualArm_v1,py`，该文件定义的环境为`SpaceRobotDualArm_v1`，需要在`SpaceRobotEnv/__init__.py`和`SpaceRobotEnv/envs/__init__.py`里添加该文件。
主要修改的内容：
将`SpaceRobotDualArm_v1`中的`initial_qpos`改为我们的空间机器人的所有关节的初始位置，然后将其中的某些参考关节替换成自己的即可，比如说将原来的`tip_frame`改为机器人末端执行器的关节（也可能是夹爪的关节，具体等仿真时再做修改）

