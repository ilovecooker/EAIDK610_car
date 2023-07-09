/**
* @par Copyright (C): 2010-2019, Shenzhen Yahboom Tech
* @file         CarRun.c
* @author       Danny
* @version      V1.0
* @date         2017.07.25
* @brief       小车前进后退左右综合实验
* @details
* @par History  见如下说明
*
*/
//定义引脚
int Left_motor_go = 4;        //左电机前进 AIN1
int Left_motor_back = 2;      //左电机后退 AIN2

int Right_motor_go = 7;       //右电机前进 BIN1
int Right_motor_back = 8;     //右电机后退 BIN2

int Left_motor_pwm = 5;       //左电机控速 PWMA
int Right_motor_pwm = 6;      //右电机控速 PWMB

int input1=A0;
int input2=A1;
int input3=A2;
int input4=A3;

int flag=1;
int temp=0;
int pwm_left;
int pwm_right;
String pwm;
int pwm_intnum[2];
int *p;

/**
* Function       setup
* @author        Danny
* @date          2017.07.25
* @brief         初始化配置
* @param[in]     void
* @retval        void
* @par History   无
*/
void setup()
{
  Serial1.begin(56000); //设置串口波特率9600
  //初始化电机驱动IO口为输出方式
  pinMode(Left_motor_go, OUTPUT);
  pinMode(Left_motor_back, OUTPUT);
  pinMode(Right_motor_go, OUTPUT);
  pinMode(Right_motor_back, OUTPUT);
 
  pinMode(input1,INPUT);
  pinMode(input2,INPUT);
  pinMode(input3,INPUT);
  pinMode(input4,INPUT);
  
  
}

/**
* Function       run
* @author        Danny
* @date          2017.07.25
* @brief         小车前进
* @param[in]     time
* @param[out]    void
* @retval        void
* @par History   无
*/
void runcar(int pwm_l,int pwm_r)
{

  //左电机前进
  digitalWrite(Left_motor_go, HIGH);   //左电机前进使能
  digitalWrite(Left_motor_back, LOW);  //左电机后退禁止
  analogWrite(Left_motor_pwm, pwm_l);    //PWM比例0-200调速，左右轮差异略增减，数值超过200将会导致小车异常断电

  //右电机前进
  digitalWrite(Right_motor_go, HIGH);  //右电机前进使能
  digitalWrite(Right_motor_back, LOW); //右电机后退禁止
  analogWrite(Right_motor_pwm, pwm_r);   //PWM比例0-200调速，左右轮差异略增减，数值超过200将会导致小车异常断电

}

/**
* Function       brake
* @author        Danny
* @date          2017.07.25
* @brief         小车刹车
* @param[in]     time
* @param[out]    void
* @retval        void
* @par History   无
*/
void brake(int pwm_l,int pwm_r)
{
  digitalWrite(Left_motor_go, LOW);
  digitalWrite(Left_motor_back, LOW);
  analogWrite(Left_motor_pwm, pwm_l);    //PWM比例0-200调速，左右轮差异略增减，数值超过200将会导致小车异常断电
  digitalWrite(Right_motor_go, LOW);
  digitalWrite(Right_motor_back, LOW);
  analogWrite(Right_motor_pwm, pwm_r);   //PWM比例0-200调速，左右轮差异略增减，数值超过200将会导致小车异常断电
}

/**
* Function       left
* @author        Danny
* @date          2017.07.25
* @brief         小车左转 左转(左轮不动,右轮前进)
* @param[in]     time
* @param[out]    void
* @retval        void
* @par History   无
*/
void left(int pwm_l,int pwm_r)
{
  //左电机停止
  digitalWrite(Left_motor_go, LOW);     //左电机前进禁止
  digitalWrite(Left_motor_back, LOW);   //左电机后退禁止
  analogWrite(Left_motor_pwm,pwm_l);       //左边电机速度设为0(0-200)

  //右电机前进
  digitalWrite(Right_motor_go, HIGH);  //右电机前进使能
  digitalWrite(Right_motor_back, LOW); //右电机后退禁止
  analogWrite(Right_motor_pwm, pwm_r);   //右边电机速度设120(0-200)

}

/**
* Function       right
* @author        Danny
* @date          2017.07.25
* @brief         小车右转 右转(左轮前进,右轮不动)
* @param[in]     time
* @param[out]    void
* @retval        void
* @par History   无
*/
void right(int pwm_l,int pwm_r)
{
  //左电机前进
  digitalWrite(Left_motor_go, HIGH);    //左电机前进使能
  digitalWrite(Left_motor_back, LOW);   //左电机后退禁止
  analogWrite(Left_motor_pwm, pwm_l);     //左边电机速度设120(0-200)

  //右电机停止
  digitalWrite(Right_motor_go, LOW);    //右电机前进禁止
  digitalWrite(Right_motor_back, LOW);  //右电机后退禁止
  analogWrite(Right_motor_pwm, pwm_r);      //右边电机速度设0(0-200)

}

/**
* Function       spin_left
* @author        Danny
* @date          2017.07.25
* @brief         小车原地左转 原地左转(左轮后退，右轮前进)
* @param[in]     time
* @param[out]    void
* @retval        void
* @par History   无
*/
void spin_left(int pwm_l,int pwm_r)
{
  //左电机后退
  digitalWrite(Left_motor_go, LOW);     //左电机前进禁止
  digitalWrite(Left_motor_back, HIGH);  //左电机后退使能
  analogWrite(Left_motor_pwm, pwm_l);

  //右电机前进
  digitalWrite(Right_motor_go, HIGH);  //右电机前进使能
  digitalWrite(Right_motor_back, LOW); //右电机后退禁止
  analogWrite(Right_motor_pwm, pwm_r);

}

/**
* Function       spin_right
* @author        Danny
* @date          2017.07.25
* @brief         小车原地右转 原地右转(右轮后退，左轮前进)
* @param[in]     time
* @param[out]    void
* @retval        void
* @par History   无
*/
void spin_right(int pwm_l,int pwm_r)
{
  //左电机前进
  digitalWrite(Left_motor_go, HIGH);    //左电机前进使能
  digitalWrite(Left_motor_back, LOW);   //左电机后退禁止
  analogWrite(Left_motor_pwm, pwm_l);

  //右电机后退
  digitalWrite(Right_motor_go, LOW);    //右电机前进禁止
  digitalWrite(Right_motor_back, HIGH); //右电机后退使能
  analogWrite(Right_motor_pwm, pwm_r);

}

/**
* Function       back
* @author        Danny
* @date          2017.07.25
* @brief         小车后退 
* @param[in]     time
* @param[out]    void
* @retval        void
* @par History   无
*/
void back(int pwm_l,int pwm_r)
{
  //左电机后退
  digitalWrite(Left_motor_go, LOW);     //左电机前进禁止
  digitalWrite(Left_motor_back, HIGH);  //左电机后退使能
  analogWrite(Left_motor_pwm, pwm_l);

  //右电机后退
  digitalWrite(Right_motor_go, LOW);    //右电机前进禁止
  digitalWrite(Right_motor_back, HIGH); //右电机后退使能
  analogWrite(Right_motor_pwm, pwm_r);


}

int pwm_input(String pwm1,int a[])
{
  int j=0;
  for(int i=0;i<pwm1.length();i++)
  {
    if(pwm1[i]==',')
    {
      j++;
    }
    else
    {
      a[j]=a[j]*10+(int(pwm1[i])-int('0'));
    }
  }
}
      
/**
* Function       loop
* @author        Danny
* @date          2017.07.25
* @brief         先延时2，再前进1，后退1s,左转2s,右转2s,
*                原地左转3s,原地右转3s,停止0.5s
* @param[in]     void
* @retval        void
* @par History   无
*/

void loop()
{
  while (Serial1.available() > 0)//串口接收到数据
  {
    pwm+=char(Serial1.read());//获取串口接收到的数据
    delay(10);
    temp=1;
  }
  
  if (temp==1)
  {
    pwm_intnum[0]=0;
    pwm_intnum[1]=0;
    
    p=pwm_intnum;
    pwm_input(pwm,p);
    pwm="";
    temp=0;
    pwm_left=pwm_intnum[0];
    pwm_right=pwm_intnum[1];
  }
  
  if (digitalRead(input4) ==HIGH)
  {
    if(digitalRead(input1) == LOW && digitalRead(input2) == LOW && digitalRead(input3) == HIGH)
    {
      runcar(pwm_left,pwm_right);          //前进1s(10 * 100ms)
    }
    else if (digitalRead(input1) == LOW && digitalRead(input2) == HIGH && digitalRead(input3) == LOW)
    {
      back(pwm_left,pwm_right);         //后退1s
    }
    else if (digitalRead(input1) == HIGH && digitalRead(input2) == LOW && digitalRead(input3) == LOW)
    {
      left(pwm_left,pwm_right);         //左转2s
    }
    else if (digitalRead(input1) == HIGH && digitalRead(input2) == LOW && digitalRead(input3) == HIGH)
    {
      right(pwm_left,pwm_right);         //后退1s
    }
    else if (digitalRead(input1) == HIGH && digitalRead(input2) == HIGH && digitalRead(input3) == LOW)
    {
      spin_left(pwm_left,pwm_right);    //原地左转3s
    }
    else if (digitalRead(input1) == HIGH && digitalRead(input2) == HIGH && digitalRead(input3) == HIGH)
    {
      spin_right(pwm_left,pwm_right);    //原地左转3s
    }
    else
    {
      brake(pwm_left,pwm_right);
    }
  }
  else
  {
    brake(pwm_left,pwm_right);         //停止0.5s
  }
}
