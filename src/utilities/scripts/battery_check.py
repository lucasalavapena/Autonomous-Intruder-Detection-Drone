#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Float32

timer = time.time()
LOW_BATTERY = 50 # percent
CRITICAL_BATTERY = 30 # percent

CRITICAL_BATTERY_MSG = """\n####################################\n###### BATTERY LEVEL CRITICAL ######\n######  BATTERY IS AT {}%  ######\n####################################\n"""
LOW_BATTERY_MSG = """\n####################################\n######   BATTERY LEVEL LOW    ######\n######  BATTERY IS AT {}%  ######\n####################################\n"""
NORMAL_BATTERY_MSG = "BATTERY LEVEL GOOD, CURRENTLY AT {}%"

def callback(data):
    global timer
    battery = (data.data - 3.0) / (4.23 - 3.0) * 100

    if battery < CRITICAL_BATTERY:
        rospy.logfatal(CRITICAL_BATTERY_MSG.format(round(battery, 2)))
    elif battery < LOW_BATTERY:
        rospy.logwarn(LOW_BATTERY_MSG.format(round(battery, 2)))
    elif time.time() - timer > 15:
        timer = time.time()
        rospy.loginfo(NORMAL_BATTERY_MSG.format(round(battery, 2)))



def listener():
    rospy.init_node('battery_check', anonymous=True)
    
    rospy.Subscriber("/cf1/battery", Float32, callback)

    rospy.spin()


if __name__ == '__main__':
    listener()
