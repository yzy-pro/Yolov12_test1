# Comsen
## 华中科技大学光学与电子信息学院2409陈恪瑾
### 项目概述
#### 项目名称
基于YOLO的目标识别与追踪
#### 数据来源
* 训练数据集：COCO[coco.yaml](coco.yaml)
* 展示数据集：[mytest](mytest)
#### 使用模型
YOLOv12

#### requirements
[requirements.txt](requirements.txt)

#### 主要参考文献（代码参考）
* https://github.com/ultralytics

### 项目成果
* 调用[main.py](main.py)的display函数可以对视频进行目标识别与追踪，并在视频中标注出目标的位置和轨迹。
* 调用[main.py](main.py)的display_webcam函数可以对摄像头进行目标识别与追踪，并在摄像头中标注出目标的位置和轨迹。



 