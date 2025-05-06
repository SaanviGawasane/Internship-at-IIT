#!/usr/bin/env python

import rospy
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped

current_state = None

def state_cb(msg):
    global current_state
    current_state = msg

def main():
    rospy.init_node("offboard_takeoff")

    state_sub = rospy.Subscriber("mavros/state", State, state_cb)
    pose_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

    rospy.wait_for_service("/mavros/cmd/arming")
    rospy.wait_for_service("/mavros/set_mode")

    arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
    set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    rate = rospy.Rate(20)

    pose = PoseStamped()
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 2  # constant altitude

    # Send some setpoints before starting OFFBOARD
    for _ in range(100):
        pose_pub.publish(pose)
        rate.sleep()

    rospy.loginfo("Setting mode to OFFBOARD...")
    set_mode_client(custom_mode="OFFBOARD")

    rospy.loginfo("Arming the drone...")
    arming_client(True)

    rospy.loginfo("Drone armed and OFFBOARD mode set!")

    while not rospy.is_shutdown():
        pose_pub.publish(pose)
        rate.sleep()

if __name__ == "__main__":
    main()
