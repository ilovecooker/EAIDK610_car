"""
预备驱动小车方案：姿态估计+DTW算法的动作识别
"""

# 导入相关包
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

import glob
import tqdm
import os
import time
from fastdtw import fastdtw
import os, sys
import getopt
import time
from enum import Enum
import keyboard
import signal
import serial
#from photo import image_white_balance,undistort_img,undistort
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

class HumanKeypoints:
    """
    获取人体Pose关键点
    """
    def __init__(self):
        
        # 加载movenet关键点检测模型
        self.interpreter = tflite.Interpreter(model_path="./Poseweights/hub/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def getFramePose(self,image):
        """
        获取关键点
        """
        # 转为RGB
        img_cvt = cv2.resize(image,(192,192))
        img_cvt = cv2.cvtColor(img_cvt,cv2.COLOR_BGR2RGB)
        img_input = img_cvt.reshape(1,192,192,3)


        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)

        self.interpreter.invoke()

        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 1,1,17,3 
        #  [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]).
        # 只需要以下
        # left shoulder: 5
        # left shoulder:6
        # left elbow: 7
        # right elbow: 8
        # left wrist: 9
        # right wrist: 10
        #
        keypoints_with_scores = keypoints_with_scores[0][0][5:11]

        return keypoints_with_scores

    def getVectorsAngle(self,v1,v2):
        """
        获取两个向量的夹角，弧度
        cos_a = v1.v2 / |v1||v2|
        """
        if np.array_equal(v1,v2):
            return 0
        dot_product = np.dot(v1,v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)


        return np.arccos(dot_product/norm) 

        

    def getFrameFeat(self,frame):
        """
        获取特征
        1.获取17个关键点，取其中6个点
        2.计算线条之间的角度信息
        3.角度信息拼成一个特征向量
        """
        # 获取关键点
        pose_landmarks = self.getFramePose(frame)
        # 连接信息
        new_conn = [(0,1),(1,3),(3,5),(0,2),(2,4)]
       
        # 解析关键点
        p_list = [[landmark[1],landmark[0]] for landmark in pose_landmarks ]
        # 转为numpy，才能广播计算
        p_list = np.asarray(p_list)
        
        # 构造向量
        # conns关键点索引之间的连接关系，利用它来构造向量（终点坐标-起点坐标）
        vector_list = list(map(
            lambda con: p_list[con[1]] - p_list[con[0]],
            new_conn
        ))
        # 计算向量之间的角度，任意两个之间夹角
        """
        !warning: 此处应该可以优化，减少特征数量
        """
        angle_list = []
        for vect_a in vector_list:
            for vect_b in vector_list:
                angle = self.getVectorsAngle(vect_a,vect_b)
                angle_list.append(angle)
        return angle_list, pose_landmarks
        


class VideoFeat:
    """
    计算特征
    计算每一帧的特征
    计算每个视频的特征
    """
    def __init__(self):
        # 加载关键点检测模型
        self.human_keypoints = HumanKeypoints()
        # 加载动作训练集特征
        self.training_feat = self.load_training_feat()

        # 批次及阈值
        self.batch_size = 4
        self.threshold = 0.5



    def get_video_feat(self,filename):
        """
        读取单个视频，获取特征
        params:
            filename: str 文件名
        """
        cap = cv2.VideoCapture(filename)
        
        # 视频特征
        video_feat = []
        while True:
            ret,frame = cap.read()
            if frame is None:
                break
            
            # 获取该帧特征
            angle_list,results = self.human_keypoints.getFrameFeat(frame)

            # 追加
            video_feat.append(angle_list)

           
            # cv2.imshow('demo',frame)
        # 保存视频特征
        return video_feat

    def load_training_feat(self):
        """
        返回训练集的特征
        如果没有，则重新生成（读取所有训练数据集，存储为npz文件）
        """
        dataFile = './data/trainingData.npz'

        if os.path.exists(dataFile):
            with open(dataFile,'rb') as f:
                return np.load(f,allow_pickle='TRUE')
            
        filename = r'./data/action_train/*/*.mp4'
        file_list = glob.glob(filename)
        training_feat = []
        for file in tqdm.tqdm(file_list,desc='训练数据集处理中') :
            action_name = file.split('/')[3]
            video_feat = self.get_video_feat(file)
            training_feat.append([action_name,video_feat])

        # 转为numpy 数组
        training_feat = np.array(training_feat,dtype=object)
        # 写入文件
        with open(dataFile,'wb') as f:
            np.save(f,training_feat)
        
        return training_feat


    def calSimilarity(self,seqFeat):
        """
        计算序列特征与训练集之间的DTW距离
        给出最终预测动作名称
        """
        # 遍历训练集中特征
        dist_list = []
        for v_feat in self.training_feat:

            action_name,video_feat = v_feat
            distance, path = fastdtw(seqFeat, video_feat)
            dist_list.append([action_name,distance])

        # 转为numpy
        dist_list = np.array(dist_list,dtype=object)
        
        # 距离由低到高排序，并截取前batch_size个
        dist_list = dist_list[dist_list[:,1].argsort()][:self.batch_size]

        print(dist_list)

        # 获取排名第一的名称
        first_key = dist_list[0][0]

        # 计算该名称出现次数
        max_num = np.count_nonzero(dist_list[:,0] == first_key)

        # 计算排序第一个的，出现总数是否超过阈值
        if max_num / self.batch_size >= self.threshold:
            print('预测动作：{}，出现次数{}/{}'.format(first_key,max_num,self.batch_size))
            # 预测动作为：first_key 
            # 对动作进行判断，并作出相应的小车动作
            if first_key == 'right':
                right()# 小车启动向右转并行动
            if first_key == 'left':
                left()# 小车启动向左转并行动
            if first_key == 'shut':
                brake()# 小车停止不动
            if first_key == 'up':
                run()# 小车向前进
            if first_key == 'dowm':
                back()# 小车向后退
            if first_key =='click':
                spin_left()# 小车原地打转
            return first_key
        else:
            print('未定义动作')
            return 'unknown'



    def realTimeVideo(self):
        """
        实时视频流动作识别
        """
        cap = cv2.VideoCapture(0)

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 录制状态
        record_status = False
        # 帧数
        frame_count = 0
        # 序列特征
        seq_feats = []

        triger_time = time.time()
        start_time = triger_time

        last_action = ''

        while True:
            key_contral()
            ret,frame = cap.read()
            if frame is None :
                break

            # 获取该帧特征
            angle_list,results = self.human_keypoints.getFrameFeat(frame)

            # 读取蓝牙串口信号
            if ser.is_open:
                len_return_data = ser.inWaiting()  # 获取缓冲数据（接收数据）长度
        # print('try to get the temperatur...')
                if len_return_data:
                    data_1 = ser.read(len_return_data)
                else:
                    data_1 = 0
            if record_status:
                # 按R后等待3秒再识别
                if time.time() - triger_time >= 1:
                
                    if frame_count < 40:
                        # < 50 帧，录制动作
                        # 录制中红色
                        cv2.circle(frame,(50,50),20,(0,255,0),-1)
                        seq_feats.append(angle_list)
                            
                        frame_count +=1
                        # print('录制中'+str(frame_count))
                    else:
                        # > 50，停止，预测
                        last_action = self.calSimilarity(seq_feats)
                        
                        # 重置
                        # record_status = False
                        frame_count = 0
                        seq_feats = []

                        record_status = True
                        triger_time = time.time()
                        print('start')
                else:
                    # 黄色3秒准备
                    cv2.circle(frame,(50,50),20,(0,255,0),-1)
            else:
                # 红色，等待
                cv2.circle(frame,(50,50),20,(0,255,0),-1)
            # 显示
            for y,x,score in results:
                x = int(x * frame_w)
                y = int(y * frame_h)
                cv2.circle(frame,(x,y),10,(0,255,0),-1)


            text = 'Pred: ' + last_action
            cv2.putText(frame,text,(50,150),cv2.FONT_ITALIC,1,(0,255,0),2)

            now = time.time()
            fps_time = now - start_time
            start_time = now

            fps_txt =round( 1/fps_time,2)
            cv2.putText(frame, str(fps_txt), (50,200), cv2.FONT_ITALIC, 1, (0,255,0),2)

            cv2.imshow('demo',frame)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  
                # 开始录制
                record_status = True
                triger_time = time.time()
                print('start')
                pass
            elif pressedKey == ord("q"):  
                break
            if data_1 == b'4':
                break
if __name__ =='__main__':
#小车驱动程序开始初始化
    print("————————小车驱动初始化开始——————————")
    ser = serial.Serial('/dev/ttyUSB0', 56000)
    ser_blue = serial.Serial('/dev/ttyUSB2',9600)
    gpio_close(pin)
    gpio_open(pin)
    brake()
    # 巡检功能不用开启
    #Tracing_setup()
    print("初始化完成")
    video = VideoFeat()  
    video.realTimeVideo()