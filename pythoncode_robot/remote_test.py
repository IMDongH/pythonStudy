import tkinter
import time

win = tkinter.Tk()

import RPi.GPIO as GPIO

import YB_Pcb_Car  # Import Yahboom car library

car = YB_Pcb_Car.YB_Pcb_Car()

num = 0
speed = 30
spin_speed = 53
repeat_delay = 30
repeat_interval = 5

Buzzer = 32  # Define the pin of the buzzer
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(32, GPIO.OUT)


def buzzerOn():
    p = GPIO.PWM(32, 440)
    p.start(50)


def forward():
    car.Car_Run(speed, speed)
    time.sleep(0.3)
    car.Car_Stop()


def back():
    car.Car_Back(speed, speed)
    time.sleep(0.3)
    car.Car_Stop()


def spin_right():
    car.Car_Spin_Right(spin_speed, spin_speed)
    time.sleep(0.15)
    car.Car_Stop()


def spin_left():
    car.Car_Spin_Left(spin_speed, spin_speed)
    time.sleep(0.15)
    car.Car_Stop()


forward_button = tkinter.Button(win, text='forward', overrelief="solid", width=15, command=forward,
                                repeatdelay=repeat_delay,
                                repeatinterval=repeat_interval)
forward_button.pack()

right_button = tkinter.Button(win, text='spin_right', overrelief="solid", width=15, command=spin_right,
                              repeatdelay=repeat_delay,
                              repeatinterval=repeat_interval)
right_button.pack()

left_button = tkinter.Button(win, text='spin_left', overrelief="solid", width=15, command=spin_left,
                             repeatdelay=repeat_delay,
                             repeatinterval=repeat_interval)
left_button.pack()

back_button = tkinter.Button(win, text='back', overrelief="solid", width=15, command=back, repeatdelay=repeat_delay,
                             repeatinterval=repeat_interval)
back_button.pack()

buzzer_button = tkinter.Button(win, text='buzzer', overrelief="solid", width=15, command=buzzerOn,
                               repeatdelay=repeat_delay,
                               repeatinterval=repeat_interval)
buzzer_button.pack()

GPIO.cleanup()
car.Car_Stop()
win.mainloop()
