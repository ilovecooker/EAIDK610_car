"""
口罩规范巡检
"""
# 导入相关包
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
# from scipy.special import softmax
import time
import sys
from fdlite.face_detection import FaceDetection, FaceDetectionModel

import os, sys
import getopt
import time
from enum import Enum
import keyboard
import signal
import serial
"""
"""
# 引脚设置部分/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 小车运动系统引脚定义
A0 = 152
A1 = 54
A2 = 56
A3 = 154
# 红外循迹系统引脚定义
left1= 66
left2 = 68
right1 = 67
right2 = 76

# 超声波接口定义
out_pin = 70
in_pin = 69

# 引脚集合
pin = [A0, A1, A2, A3, left1, left2, right1, right2,out_pin, in_pin]

#from photo import image_white_balance,undistort_img,undistort
#  定义引脚的输入输出模式
class DIRECTION(Enum):
    INVALI_DIRECTION = 0
    INPUT = 1
    OUTPUT = 2


#定义引脚中断下的触发模式
class EDGE(Enum):
    INVALI_EDGE = 0
    NONE = 1
    RISING = 2
    FALLING = 3
    BOTH = 4


# 初始化设置函数部分/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#引脚激活函数
def gpio_export(gpio):
    cmd = 'echo {} >/sys/class/gpio/export'.format(gpio)
    os.system(cmd)


#引脚批量激活
def gpio_open(pin_list):
    for i in pin_list:
        gpio_export(i)


#对引脚输出输入模式，高低电平设置
def gpio_init(gpio, direction, edge, value):

    if (direction == DIRECTION.INPUT):
        file = '/sys/class/gpio/gpio{}/direction'.format(gpio)
        with open(file, 'w') as fd_tmp:
            fd_tmp.write('in')
    elif (direction == DIRECTION.OUTPUT):
        file = '/sys/class/gpio/gpio{}/direction'.format(gpio)
        cmd = 'echo out >' + file
        os.system(cmd)

    if (edge == EDGE.NONE):
        file = '/sys/class/gpio/gpio{}/edge'.format(gpio)
        with open(file, 'w') as fd_tmp:
            fd_tmp.write('none')

    if (edge == EDGE.RISING):
        file = '/sys/class/gpio/gpio{}/edge'.format(gpio)
        with open(file, 'w') as fd_tmp:
            fd_tmp.write('rising')

    if (edge == EDGE.FALLING):
        file = '/sys/class/gpio/gpio{}/edge'.format(gpio)
        with open(file, 'w') as fd_tmp:
            fd_tmp.write('falling')

    if (edge == EDGE.BOTH):
        file = '/sys/class/gpio/gpio{}/edge'.format(gpio)
        with open(file, 'w') as fd_tmp:
            fd_tmp.write('both')

    if (value != None):
        value_file = '/sys/class/gpio/gpio{}/value'.format(gpio)
        with open(value_file, 'w') as value_file:
            value_file.write(str(value))


#引脚高低电平值读取
def get_voltage(fd_path):
    with open(fd_path, 'r') as fd:
        fd.seek(0, 0)
        return fd.read()

#引脚资源释放函数
def gpio_release(gpio):
    cmd = 'echo {} >/sys/class/gpio/unexport'.format(gpio)
    os.system(cmd)

#引脚资源批量释放函数
def gpio_close(pin):
    for i in pin:
        file_path = "/sys/class/gpio/gpio{}".format(str(i))
        if os.path.exists(file_path):
            gpio_release(i)

#小车运动方式定义函数部分///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#刹车
def brake(t=0):
    gpio_init(A0, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 0)
    gpio_init(A1, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 0)
    gpio_init(A2, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 0)
    gpio_init(A3, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 0)
    if t != 0:
        time.sleep(t)

#前进
def run(t=0.0, pwm_left=200, pwm_right=200):
    if ser.is_open:
        # 'ABC'.encode('ascii')
        str_merge = str(pwm_left) + ',' + str(pwm_right)
        send_data = str_merge.encode('ascii')
        ser.write(send_data)  # 发送命令
        time.sleep(0.1)
    gpio_init(A0, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 0)
    gpio_init(A1, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 0)
    gpio_init(A2, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 1)
    gpio_init(A3, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 1)
    if t != 0.0:
        time.sleep(t)
        return

#后退
def back(t=0.0, pwm_left=200, pwm_right=200):
    if ser.is_open:
        # 'ABC'.encode('ascii')
        str_merge = str(pwm_left) + ',' + str(pwm_right)
        send_data = str_merge.encode('ascii')
        ser.write(send_data)  # 发送命令
        time.sleep(0.1)
    gpio_init(A0, 2, 0, 0)
    gpio_init(A1, 2, 0, 1)
    gpio_init(A2, 2, 0, 0)
    gpio_init(A3, 2, 0, 1)
    if t != 0.0:
        time.sleep(t)
        return

#向左进行
def left(t=0.0, pwm_left=0, pwm_right=150):
    if ser.is_open:
        # 'ABC'.encode('ascii')
        str_merge = str(pwm_left) + ',' + str(pwm_right)
        send_data = str_merge.encode('ascii')
        ser.write(send_data)  # 发送命令
        time.sleep(0.1)
    gpio_init(A0, 2, 0, 1)
    gpio_init(A1, 2, 0, 0)
    gpio_init(A2, 2, 0, 0)
    gpio_init(A3, 2, 0, 1)
    if t != 0.0:
        time.sleep(t)
        return

#向右前进
def right(t=0.0, pwm_left=150, pwm_right=0):
    if ser.is_open:
        # 'ABC'.encode('ascii')
        str_merge = str(pwm_left) + ',' + str(pwm_right)
        send_data = str_merge.encode('ascii')
        ser.write(send_data)  # 发送命令
        time.sleep(0.1)
    gpio_init(A0, 2, 0, 1)
    gpio_init(A1, 2, 0, 0)
    gpio_init(A2, 2, 0, 1)
    gpio_init(A3, 2, 0, 1)
    if t != 0.0:
        time.sleep(t)
        return

#原地左转
def spin_left(t=0.0, pwm_left=150, pwm_right=150):
    if ser.is_open:
        # 'ABC'.encode('ascii')
        str_merge = str(pwm_left) + ',' + str(pwm_right)
        send_data = str_merge.encode('ascii')
        ser.write(send_data)  # 发送命令
        time.sleep(0.1)
    gpio_init(A0, 2, 0, 1)
    gpio_init(A1, 2, 0, 1)
    gpio_init(A2, 2, 0, 0)
    gpio_init(A3, 2, 0, 1)
    if t != 0.0:
        time.sleep(t)
        return

#原地右转
def spin_right(t=0.0, pwm_left=150, pwm_right=150):
    if ser.is_open:
        # 'ABC'.encode('ascii')
        str_merge = str(pwm_left) + ',' + str(pwm_right)
        send_data = str_merge.encode('ascii')
        ser.write(send_data)  # 发送命令
        time.sleep(0.1)
    gpio_init(A0, 2, 0, 1)
    gpio_init(A1, 2, 0, 1)
    gpio_init(A2, 2, 0, 1)
    gpio_init(A3, 2, 0, 1)
    if t != 0.0:
        time.sleep(t)
        return


#功能设计函数部分////////////////////////////////////////////////////////////////////////////////////////
#红外循迹参数初始化函数
def Tracing_setup():
    gpio_init(left1, DIRECTION.INPUT, EDGE.INVALI_EDGE, None)
    gpio_init(left2, DIRECTION.INPUT, EDGE.INVALI_EDGE, None)
    gpio_init(right1, DIRECTION.INPUT, EDGE.INVALI_EDGE, None)
    gpio_init(right2, DIRECTION.INPUT, EDGE.INVALI_EDGE, None)

#红外循迹功能运行函数，在运行前先调用初始化函数
def Tracing():
    Tpin_list=[left1, left2, right1, right2]
    str_value=""
    for i in Tpin_list:
        Tvalue_path='/sys/class/gpio/gpio{}/value'.format(i)
        str_value=str_value+str(get_voltage(Tvalue_path)[0])
    print(str_value)
    if (str_value=="1001"):
        run(0.01,50,50)
    elif(str_value=="1011"):
        left(0.01,0,80)
    elif(str_value=="1101"):
        right(0.01,80,0)
    elif(str_value=="0001" or str_value=="0101"):
        spin_left(0.01,50,50)
    elif(str_value=="1000" or str_value=="1010"):
        spin_right(0.01,50,50)
    elif(str_value=="1100" or str_value=="1110"):
        spin_right(0.01,50,50)
    elif(str_value=="0111" or str_value=="0011"):
        spin_left(0.01,50,50)
    elif(str_value=="1111"):
        back(0.01,50,50)
    else:
        brake()

def key_contral():
    if keyboard.is_pressed("w"):
        run(0.1,50,50)
    elif keyboard.is_pressed("s"):
        back(0.1,50,50)
    elif keyboard.is_pressed("a"):
        left(0.1,0,50)
    elif keyboard.is_pressed("d"):
        right(0.1,50,0)
    elif keyboard.is_pressed("q"):
        spin_left(0.1,50,50)
    elif keyboard.is_pressed("e"):
        spin_right(0.1,50,50)
    else:
        brake()

#超声波测距参数初始化函数
def distance():
    global TEMP
    value_path = '/sys/class/gpio/gpio{}/value'.format(in_pin)
    gpio_init(in_pin, DIRECTION.INPUT, EDGE.INVALI_EDGE, None)
    gpio_init(out_pin, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 1)
    time.sleep(0.000002)
    gpio_init(out_pin, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 0)
    time.sleep(0.000010)
    gpio_init(out_pin, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 1)
    while int(get_voltage(value_path)) ==0:
        pass
    t1 = time.time()
    while int(get_voltage(value_path)) == 1:
        pass
    t2 = time.time()
    dis = int(((t2 - t1) * 340 / 2)*100)
    if dis==None:
        dis=255.0
    return dis

#超声波测距运行函数，返回值为障碍物距离/cm
def distance_cpu():
    d=[]
    i=0
    while i<5:
        dis=distance()
        if dis==None:
            continue
        else:
            d.append(dis)
            i=i+1
    max_d=max(d)
    min_d=min(d)
    d.remove(max_d)
    d.remove(min_d)
    dis_avr=sum(d)//len(d)
    print("distance is {}cm".format(dis_avr))
    return dis_avr

class MaskDetection:

    """
    口罩检测：正常、未佩戴、不规范（漏鼻子）
    可运行在树莓派
    """

    def __init__(self,mode='rasp'):
        """
        构造函数
        """
        # 加载人脸检测模型
        
        # 加载口罩模型

        self.interpreter = tflite.Interpreter(model_path="./data/model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 标签
        self.labels = ['正常','未佩戴','不规范']
        # 颜色，BGR顺序，绿色、红色、黄色
        self.colors = [(0,255,0),(0,0,255),(0,255,255)]

        # 中文label图像
        self.zh_label_img_list = self.getPngList()


    def getPngList(self):
        """
        获取PNG图像列表

        @return numpy array list
        """
        overlay_list = []
        # 遍历文件
        for i in range(3):
            fileName = './label_img/%s.png' % (i)
            overlay = cv2.imread(fileName,cv2.COLOR_RGB2BGR)
            overlay = cv2.resize(overlay,(0,0), fx=0.3, fy=0.3)
            overlay_list.append(overlay)

        return overlay_list


    
    def imageProcess(self,face_region):
        """
        将图像转为blob

        @param: face_region numpy arr 输入的numpy图像
        @return: blob或None 
        """
        
        if face_region is not None:
            # blob处理
            blob = cv2.dnn.blobFromImage(face_region,1,(100,100),(104,117,123),swapRB=True)
            blob_squeeze = np.squeeze(blob).T
            blob_rotate = cv2.rotate(blob_squeeze,cv2.ROTATE_90_CLOCKWISE)
            blob_flip = cv2.flip(blob_rotate,1)
            blob_norm = np.maximum(blob_flip,0) / blob_flip.max()
            # face_resize = cv2.resize(face_region,(100,100))
            return blob_norm
        else:
            return None

    def detect(self):
        """
        识别
        """

        # 人脸检测器
        detect_faces = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA)
        
        # 获取视频流
        cap = cv2.VideoCapture(0)

        # 视频宽度和高度
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



        # 记录帧率时间
        frameTime = time.time()


        
        #time.sleep(5)
        while True:
            #开启感知车自动巡检功能
            Tracing()
            
            # 读取
            ret,frame = cap.read()
            # 翻转
            frame = cv2.flip(frame,1)
            # 检测
            faces = detect_faces(frame)
            # 操纵小车
            
            # # 读取蓝牙串口信号
            print("————————蓝牙串口读取—————————")
            if ser.is_open:
                len_return_data = ser.inWaiting()  # 获取缓冲数据（接收数据）长度
        # print('try to get the temperatur...')
                if len_return_data:
                    data_1 = ser.read(len_return_data)
                else:
                    data_1 = 0
            
            # 记录人数
            person_count = 0

            if not len(faces):
                print('no faces detected :(')
            else:
                
                # 遍历多个人脸
                for face in faces:

                    person_count+=1

                    l,t,r,b = ([face.bbox.xmin,face.bbox.ymin,face.bbox.xmax,face.bbox.ymax] * np.array([frame_w,frame_h,frame_w,frame_h])).astype(int)
                    
                    
                    t -= 20
                    b += 20
                    # 越界处理
                    if l<= 0 or t <=0 or r >= frame_w or b >= frame_h :
                        continue

                    # 人脸区域
                    face_region = frame[t:b,l:r]
                    # 转为blob
                    blob_norm = self.imageProcess(face_region)

                    if blob_norm is not None:
                        # 预测
                        img_input = blob_norm.reshape(1,100,100,3)

                        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)

                        self.interpreter.invoke()

                        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                        result = output_data
                        # 最大值索引
                        # softmax处理
                        # result = softmax(result)[0]
                        # 最大值索引
                        max_index = result[0].argmax()
                        # 最大值
                        # max_value = result[0][max_index]
                        # 标签
                        label = self.labels[max_index]
                        #print("是否佩戴口罩")
                        #print(label)
                        #labels = ['正常','未佩戴','不规范']
                        # 检测到口罩规范情况，嵌入式做出相应动作
                        if label =='正常':
                            # 小车停车3s,语音播报
                            print("检测到口罩佩戴正常")
                            playsound('./voices/kouzhao_ok.mp3')
                        if label == '未佩戴':
                            # 小车停车3s,并语音播报
                            print("检测到口罩未佩戴")
                            playsound('./voices/kouzhao_no.mp3')
                        if label == '不规范':
                            # 小车停车3s,并语音播报
                            print("检测到口罩佩戴不规范")
                            playsound('./voices/kouzhao_notgood.mp3')

                        # 中文标签
                        overlay = self.zh_label_img_list[max_index]
                        overlay_h,overlay_w = overlay.shape[:2]

                        # 覆盖范围
                        overlay_l,overlay_t = l,(t - overlay_h-20)
                        overlay_r,overlay_b = (l + overlay_w),(overlay_t+overlay_h)

                        # 判断边界
                        if overlay_t > 0 and overlay_r < frame_w:
                            
                            overlay_copy=cv2.addWeighted(frame[overlay_t:overlay_b, overlay_l:overlay_r ],1,overlay,20,0)
                            frame[overlay_t:overlay_b, overlay_l:overlay_r ] = overlay_copy

                            # cv2.putText(frame, str(round(max_value*100,2))+"%", (overlay_r+20, overlay_t+40), cv2.FONT_ITALIC, 0.8, self.colors[max_index], 2)

                    # 人脸框
                    cv2.rectangle(frame,(l,t),(r,b),self.colors[max_index],5)


            now = time.time()
            fpsText = 1 / (now - frameTime)
            frameTime = now

            cv2.putText(frame, "FPS:  " + str(round(fpsText,2)), (50, 60), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, "Person:  " + str(person_count), (50, 110), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 3)


            cv2.imshow('demo',frame)
            
            if cv2.waitKey(1) & 0xFF == ord('m'): #or data_1 == b"4":
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ =='__mian__':
    print("————————小车驱动初始化开始——————————")
    ser = serial.Serial('/dev/ttyUSB0', 56000)
    ser_blue = serial.Serial('/dev/ttyUSB2',9600)
    gpio_close(pin)
    gpio_open(pin)
    brake()
    Tracing_setup()
    print("初始化完成")
    # 调用检测函数
    mask_detection = MaskDetection()
    mask_detection.detect()