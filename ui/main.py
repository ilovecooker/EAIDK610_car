"""
客户端运行文件

"""

from car import Ui_Form
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets
import sys
import serial


class mainEntry(QMainWindow,Ui_Form,QtWidgets.QWidget):
    def __init__(self):
        super(mainEntry,self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.back)
        self.pushButton_3.clicked.connect(self.right)
        self.pushButton_4.clicked.connect(self.left)
        self.pushButton_5.clicked.connect(self.stop)
        self.pushButton_6.clicked.connect(self.run)
        self.pushButton_7.clicked.connect(self.move_contral)
        self.pushButton_8.clicked.connect(self.people)
        self.pushButton_9.clicked.connect(self.mask)
        self.pushButton_10.clicked.connect(self.gesture)
        self.pushButton_11.clicked.connect(self.close)
        self.ser = serial.Serial('COM24', 9600)
        self.contral_flag = 0
        self.mode_flag=0
        self.data = 0
        self.len_return_data=0
        self.face_name=object
        self.face_user=object
        self.face_tangles=object
        self.face_list=object
        self.path2=object
    def bluetooth(self,a="0"):
        # 创建对象
        # 选择串口，并设置波特率
        if self.ser.is_open:
            # 'ABC'.encode('ascii')
            data=a
            send_data = data.encode('ascii')
            self.ser.write(send_data)  # 发送命令
    def bluetooth_get(self):
        if self.ser.is_open:
            self.len_return_data = self.ser.inWaiting()  # 获取缓冲数据（接收数据）长度
            # print('try to get the temperatur...')
            if self.len_return_data:
                self.data = self.ser.read(self.len_return_data)
    def run(self):
        if self.contral_flag==1:
            self.bluetooth("run")
            self.lineEdit_2.setText("提示：前进")
        return
    def left(self):
        if self.contral_flag==1:
            self.bluetooth("left")
            self.lineEdit_2.setText("提示：左拐")
        return
    def right(self):
        if self.contral_flag==1:
            self.bluetooth("right")
            self.lineEdit_2.setText("提示：右拐")
        return
    def back(self):
        if self.contral_flag==1:
            self.bluetooth("back")
            self.lineEdit_2.setText("提示：后退")
        return
    def stop(self):
        self.contral_flag=0
        self.lineEdit.setText("状态提醒：远程控制已经关闭")
        return
    def people(self):
        self.mode_flag = 1
        self.bluetooth("1")
        self.lineEdit.setText("状态提醒：开启智能巡检行人安全距离，点击关闭按钮关闭")
        return
    def mask(self):
        self.bluetooth("2")
        self.lineEdit.setText("状态提醒：开启智能巡检口罩规范佩戴，点击关闭按钮关闭")
        return
    def move_contral(self):
        self.contral_flag=1
        self.lineEdit.setText("状态提醒：开启"
                              "远程控制已经开启，，点击停止按钮关闭")
    def gesture(self):
        self.bluetooth("3")
        self.lineEdit.setText("状态提醒：智能手势控制")
        return
    def close(self):
        self.bluetooth("4")
        self.mode_flag=0
        self.lineEdit.setText("状态提醒：已经关闭")
        return
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mainEntry()
    window.show()
    sys.exit(app.exec_())