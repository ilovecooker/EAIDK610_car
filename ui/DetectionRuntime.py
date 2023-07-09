"""
目标检测巡检
"""
import cv2
import time
import numpy as np
import onnxruntime 
import math
import os, sys
import getopt
import time
from enum import Enum
import keyboard
import signal
import serial
from playsound import playsound
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
        
def key_contral():
    if keyboard.is_pressed("w"):
        run(0.1,100,100)
    elif keyboard.is_pressed("s"):
        back(0.1,100,100)
    elif keyboard.is_pressed("a"):
        left(0.1,0,100)
    elif keyboard.is_pressed("d"):
        right(0.1,100,0)
    elif keyboard.is_pressed("q"):
        spin_left(0.1,100,100)
    elif keyboard.is_pressed("e"):
        spin_right(0.1,100,100)
    else:
        brake()
#前进
def run(t=0.0, pwm_left=30, pwm_right=30):
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
def back(t=0.0, pwm_left=30, pwm_right=30):
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
        if dis==255.0:
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

# sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# tanh函数
def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

# 数据预处理
def preprocess(src_img, size):
    output = cv2.resize(src_img,(size[0], size[1]),interpolation=cv2.INTER_AREA)
    output = output.transpose(2,0,1)
    output = output.reshape((1, 3, size[1], size[0])) / 255

    return output.astype('float32')

# nms算法
def nms(dets, thresh=0.35): # 原始thresh = 0.35
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # #thresh:0.3,0.5....
    #print(type(dets))
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标

    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)  # keep里保存的是最高的box值

        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= thresh)[0]

        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    
    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output   # output为一个一维列表

# 检测
def detection(session, img, input_width, input_height, thresh):
    pred = [[0,0,0,0,0,0]]

    # 输入图像的原始宽高
    H, W, _ = img.shape

    # 数据预处理: resize, 1/255
    data = preprocess(img, [input_width, input_height])

    # 模型推理
    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    # 输出特征图转置: CHW, HWC
    feature_map = feature_map.transpose(1, 2, 0)
    # 输出特征图的宽高
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    # 特征图后处理
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            # 解析检测框置信度
            obj_score, cls_score = data[0], data[5:].max()
            score = obj_score * cls_score

            # 阈值筛选
            if score > thresh:
                # 检测框类别
                cls_index = np.argmax(data[5:])
                # 检测框中心点偏移
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                # 检测框归一化后的宽高
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                # 检测框归一化后中心点
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height
                
                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])
                #print(pred)
    return nms(np.array(pred))  # 通过非极大抑制值来进行处理

if __name__ == '__main__':
    #小车驱动程序开始初始化
    print("————————小车驱动初始化开始——————————")
    ser = serial.Serial('/dev/ttyUSB0', 56000)
    ser_blue = serial.Serial('/dev/ttyUSB2',9600)
    gpio_close(pin)
    gpio_open(pin)
    brake()
    Tracing_setup()
    print("初始化完成")
    # 读取图片
    capture = cv2.VideoCapture(0)
    fps_time = time.time()
    #key_contral()
    # 模型输入的宽高
    input_width, input_height =256 , 256
    # 加载模型
    session = onnxruntime.InferenceSession('./detectionModel/FastestDet256.onnx')
    # 目标检测
    # 加载mp4
    while True:
        # 开启感知车自动巡线功能
        Tracing()
        
        ret,img = capture.read()
        img = cv2.flip(img,1)

        # bboxes为一个二维列表——————>输出为:[[1,2,3,4,5,6]]
        bboxes = detection(session, img, input_width, input_height, 0.8)  
        
        # 读取蓝牙串口信号
        print("————————蓝牙串口读取—————————")
        if ser_blue.is_open:
            len_return_data = ser_blue.inWaiting()  # 获取缓冲数据（接收数据）长度
        # print('try to get the temperatur...')
            if len_return_data:
                data_1 = ser_blue.read(len_return_data)
            else:
                data_1 = 0
        
    # 加载label names
        names = []
        with open("coco.names", 'r') as f:
	        for line in f.readlines():
	            names.append(line.strip())   
        #print("=================box info===================")
        #print(bboxes)

        print("检测到{}个人".format(np.array(bboxes).shape[0]-1))
        # 判断行人距离,
        if (np.array(bboxes)).shape[0]==2:
            if bboxes[0][5] == 4:
                print("检测到一个行人")
                
                # 获取超声波传感器的距离
                chaoshengbojuli = distance_cpu()

                # 在putext画面上显示检测到人的数量
                cv2.putText(img,"people number:{}".format(np.array(bboxes).shape[0]-1),(20,50),cv2.FONT_ITALIC,1,(0,255,0),5)
                cv2.putText(img,str(round(chaoshengbojuli,2)),(20,80),cv2.FONT_ITALIC,1,(0,225,0),5)
        if (np.array(bboxes)).shape[0]==3:
            if bboxes[0][5] == bboxes[1][5] :
                print("当前检测到2个人——————进行行人距离判断")
                cv2.putText(img,"people number:{}".format(np.array(bboxes).shape[0]-1),(20,30),cv2.FONT_ITALIC,1,(0,255,0),5)
                # 计算靠近距离
                distance =math.sqrt(
                            (bboxes[0][0]-bboxes[1][0]) *(bboxes[0][0]-bboxes[1][0]) + 
                            (bboxes[0][1]-bboxes[1][1])*(bboxes[0][1]-bboxes[1][1])
                        )
                # 安全距离，使用归一化人体身高方法来判断
                anquan_distance = (bboxes[0][2])/2
                # 对安全距离判断
                if distance > anquan_distance:
                    print("——————————当前行人间符合安全距离———————————")
                    
                    #语音播报

                if distance <anquan_distance:
                    print("——————————行人间距离过近，请保持安全距离———————————")
                    playsound('./voices/people_safe.mp3')
                    # 语音播报
                    
        for b in bboxes:
            obj_score, cls_index = b[4], int(b[5])
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            
            
            # 对检测目标进行预测
            if cls_index == 1 :
                print("————————检测到 : red_light————————")
                print("检测到红色交通灯，请停车")
                
                #在检测到相关目标时，再对距离进行判断，同时满足两个条件，小车停
                if distance_cpu() <30 :
                    # 语音播报+停车
                    playsound('./voices/red_light_stop.mp3')
                    brake(3)


            if cls_index == 2:
                print("————————检测到 yellow_light ——————————")
                print("检测到黄色交通灯，请减速慢行或停车")
                if distance_cpu() <30 :
                    # 语音播报+停车
                    playsound('./voices/yellow_light_stop.mp3')
                    brake(3)
                    
            if cls_index == 3 :
                print("————————检测到 green_light ——————————")
                # 小车继续前进
                # 语音播报
                print("检测到绿色交通灯，请保持正常行驶")
                playsound('./voices/green_light.mp3')
                
            if cls_index == 4 :
                print("————————检测到 people—————————")
                # 嵌入式板通报“前方有行人，减速慢行”，并进行减速
                if distance_cpu() <30 :
                   # pass
                # 语音播报+停车
                    print("前方有行人，请减速慢行")
                    playsound('./voices/people_stop_go.mp3')
                    brake(3)    
            
            #绘制检测框
            cv2.rectangle(img, (x1,y1), (x2, y2), (255, 255, 0), 5)
            cv2.putText(img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(img, names[cls_index], (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        now = time.time()
        fps_text = 1/(now-fps_time)
        fps_time = now
        cv2.putText(img,str(round(fps_text,2)),(20,20),cv2.FONT_ITALIC,1,(0,255,0),5)
        cv2.imshow("demo",img)
        # 若按下键盘q，则退出播放
        if cv2.waitKey(20) & 0xFF == ord('m') or data_1 == b"4":
            break
    capture.release()
    cv2.destroyAllWindows()
    #cv2.imwrite("result.jpg", img)
