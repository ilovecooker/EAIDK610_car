#  基于EAIDK-610的防疫不规范行为的自动巡检系统、
## 设计框图
本次设计的智能巡检感知车属于嵌入式产品，嵌入式系统设计需要定义系统的功能、决定系统的架构，并将功能映射到系统实现架构上。这里，系统架构既包括软件系统架构也包括硬件系统架构。硬件系统设计是基于EAIDK-610作为核心开发平台，Arduino作为感知车电机驱动芯片的控制器，同时搭配一些外设传感器视频采集模块、超声避障模块和红外循迹模块，蓝牙模块等。软件方面，主要包括系统控制算法设计，电机驱动算法设计，超声传感器智能避障算法设计，口罩佩戴检测算法设计，人员安全距离检测算法设计等部分组成，以及人机交互端APP 软件设计等部分共同组成。系统总体设计框架如下图 所示：

![image](https://github.com/ilovecooker/EAIDK610_car/assets/92043804/fd21db12-7fef-4d4a-8d28-99d5a598f791)

本设计整体的系统构架从下至上依次有硬件层，驱动层，数据交互层，数据处理层，控制算法层和应用层。
（1）硬件层
硬件层是本设计最直观的物理体现形式，主要包括电机、舵机、超声波传感器、红外传感器、摄像头等。
（2）驱动层
驱动层是与硬件直接进行交互使其运行起来的的最底层软件，包括驱动电机舵机的PWM，获取超声波传感器的输入捕捉，以及用于人机，机机交互的GPIO和UART。
（3）数据交互层
数据交互层是连接底层与上层的关键一层，它是对驱动层从数据流的角度进一步的抽象和封装，为上层应用对底层硬件的访问控制提供了统一的接口和权限的控制。
（4）数据滤波层
数据滤波层是数据经过的第一层处理，主要提供了对数据最基础的数学运算变换以及数字信号处理手段。
（5）控制算法层
基于底层提供的原始数据或者处理过的数据，控制算法层将这些数据带入到实际模型中完成对模型的控制。
（6）应用层
应用层是最上层的逻辑处理层，在这一层实现最具体的功能，例如让小车循迹行驶、以及小车自主行驶策略控制等均在应用层完成。下面对本架构从硬件和软件的角度展开阐述。

## 系统开发环境搭建
### 系统硬件开发环境
硬件设计主要是设计电路原理图和绘制 PCB 电路板，这里我们釆用 Altium Designer软件设计电路板。Altium Designer完美融合了电路板的完整开发流程，包括原理图设计、电路仿真、PCB绘制编辑、拓扑逻辑自动布线、信号完整性分析和设计输出等，真正做到一体式设计，为设计者提供了一次性的设计解决方案，使设计者可以利用各种开发工具，轻松进行设计，熟练使用这一软件可以使电路设计的质量和效率大大提高。它集合了电子产品开发所需要的各种开发工具，包括了初期的设计硬件电路图、测试过程分析信号的完整性、后期印制电路板以及仿真整个系统电路等。
### 系统软件开发环境
本设计，以EAIDK610开发平台作为计算中心，以基于Linux的Fedora桌面操作系统作为开发系统，使用python作为主要开发语言。采用SSH和VNC远程终端控制软件作为开发效率工具，VSCODE代码编辑器作为开发辅助工具。以Arduino IDE作为感知车驱动和传感器逻辑推理模块的编辑软件。在计算机视觉深度学习算法方面，采用多框架部署方式，采用基于tenorflow框架的tflite推理模型部署和基于pytorch模型训练，onnx工具部署。

## 系统总体构架
硬件设计的总体构架框图如下图 所示：EAIDK610作为系统的核心，与各个设备连接成为系统信息接收，处理，发送的信息处理中心。与Arduino微型控制器通过串口与GPIO接口连接，Arduino通过输出端接口与电机驱动芯片连接，便可实现感知车的各种运动；通过蓝牙串口通信模块与Arduino连接，可实现小车运动的远程控制。将USB摄像头与EAIDK610连接，可实时接收摄像头采集图像信息；将超声传感器通过EAIDK610的GPIO接口连接可实时感知传感器采集的障碍物距离信息。使用电源模块给各模块供电。由此将视频信号与传感器信号结合，可实现感知车对核酸检测排队场景中待检测人员防疫不规范行为的巡检。

![image](https://github.com/ilovecooker/EAIDK610_car/assets/92043804/50a2293e-f6d6-4d93-ae81-fd52705896cb)
