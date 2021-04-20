#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32

LOW_BATTERY = 50 # percent
CRITICAL_BATTERY = 30 # percent

CRITICAL_BATTERY_MSG = """\n####################################\n###### BATTERY LEVEL CRITICAL ######\n######  BATTERY IS AT {}%  ######\n####################################\n"""
LOW_BATTERY_MSG = """\n####################################\n######   BATTERY LEVEL LOW    ######\n######  BATTERY IS AT {}%  ######\n####################################\n"""
NORMAL_BATTERY_MSG = "BATTERY LEVEL GOOD, CURRENTLY AT {}%"

def callback(data):
    battery = (data.data - 3.0) / (4.23 - 3.0) * 100

    if battery < CRITICAL_BATTERY:
        rospy.logfatal_throttle(0.5, CRITICAL_BATTERY_MSG.format(round(battery, 2)))
    elif battery < LOW_BATTERY:
        rospy.logwarn_throttle(3, LOW_BATTERY_MSG.format(round(battery, 2)))
    else:
        rospy.loginfo_throttle(15, NORMAL_BATTERY_MSG.format(round(battery, 2)))



def listener():
    rospy.init_node('battery_check', anonymous=True)
    
    rospy.Subscriber("/cf1/battery", Float32, callback)

    rospy.spin()


if __name__ == '__main__':
    listener()
