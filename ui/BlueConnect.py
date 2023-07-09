import serial
import os
import time
import sys
import getopt
import time
from enum import Enum
import keyboard
import signal


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
        left(0.01,0,100)
    elif(str_value=="1101"):
        right(0.01,100,0)
    elif(str_value=="0001" or str_value=="0101"):
        spin_left(0.01,50,50)
    elif(str_value=="1000" or str_value=="1010"):
        spin_right(0.01,50,50)
    elif(str_value=="1100" or str_value=="1110"):
        spin_right(0.01,50,50)
    elif(str_value=="0111" or str_value=="0011"):
        spin_left(0.01,50,50)
    else:
        brake()

#超声波测距参数初始化函数
def distance():
    global TEMP
    flag1=1
    flag2=1
    value_path = '/sys/class/gpio/gpio{}/value'.format(in_pin)
    gpio_init(in_pin, DIRECTION.INPUT, EDGE.INVALI_EDGE, None)
    gpio_init(out_pin, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 1)
    time.sleep(0.000002)
    gpio_init(out_pin, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 0)
    time.sleep(0.000015)
    gpio_init(out_pin, DIRECTION.OUTPUT, EDGE.INVALI_EDGE, 1)
    while int(get_voltage(value_path)) ==0:
        if flag1==1:
            flag1=0
            st=time.time()
        end_t=time.time()
        TEMP=end_t-st
        if TEMP>0.148:
            return None
        pass
    t1 = time.time()
    while int(get_voltage(value_path)) == 1:
        if flag2==1:
            flag2=0
        if TEMP>0.148:
            return None
        pass
    t2 = time.time()
    dis = int(((t2 - t1) * 340 / 2)*100)
    if flag1+flag2==0:
        return dis
    else:
        return None

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

if __name__ == '__main__':
    #一些初始化操作放这里
    ser = serial.Serial('/dev/ttyUSB0', 56000)#激活串口
    ser_blue = serial.Serial('/dev/ttyUSB2', 9600)#激活串口
    gpio_close(pin)
    gpio_open(pin)
    brake()
    Tracing_setup()
    print("初始化完成")

while True:
    #创建对象
    #选择串口，并设置波特率 
    if ser_blue.is_open:
        len_return_data = ser_blue.inWaiting()  # 获取缓冲数据（接收数据）长度
        # print('try to get the temperatur...')
        if len_return_data:
            data = ser_blue.read(len_return_data)
              # 读取缓冲数据
            print(data)
            try:
                if data==b"run":
                    run(0.5,50,50)
                    print(data)
                    print("前进1s")
                    brake()
                elif data==b"back":
                    back(0.5,50,50)
                    print(data)
                    print("后退1s")
                    brake()
                elif data==b"left":
                    left(0.5,0,50)
                    print(data)
                    print("左拐1s")
                    brake()
                elif data==b"right":
                    right(0.5,50,0)
                    print(data)
                    print("右拐1s")
                    brake()
                elif data==b"1":
                    print("打开程序1")
                    cmd_open = 'python3 /home/openailab/Desktop/code/1.detection_Runtime.py'
                    os.system(cmd_open)
                    print("关闭程序1")
                elif data==b"2":
                    print("打开程序2")
                    cmd_open = 'python3 /home/openailab/Desktop/code/2.Mask_identification.py'
                    os.system(cmd_open)
                    print("关闭程序2")
                elif data==b"3":
                    print("打开程序3")
                    cmd_open = 'python3 /home/openailab/Desktop/code/3.PoseCar.py'
                    os.system(cmd_open)
                    print("关闭程序3")
                elif data==b'4':
                    break
                return_data=0
                

            except:
                gpio_close(pin)
                continue
    else:
        print("port open failed")
