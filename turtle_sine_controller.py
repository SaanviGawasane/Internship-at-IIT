#!/usr/bin/env python3

import rospy
import math
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import TeleportAbsolute

class TurtleSineTracer:
    def __init__(self):
        rospy.init_node('turtle_sine_tracer', anonymous=True)

        self.vel_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)

        self.pose = Pose()
        self.rate = rospy.Rate(20)

        # Sine wave parameters
        self.amplitude = 2.0
        self.frequency = 1.5
        self.k_heading = 4.0
        self.linear_speed = 1.0

        # Position tracking
        self.actual_x = []
        self.actual_y = []
        self.desired_y = []

        # Teleport to start
        rospy.wait_for_service('/turtle1/teleport_absolute')
        self.teleport_service = rospy.ServiceProxy('/turtle1/teleport_absolute', TeleportAbsolute)
        self.teleport_turtle(1.0, 5.0, 0.0)
        rospy.sleep(1)

    def teleport_turtle(self, x, y, theta):
        try:
            self.teleport_service(x, y, theta)
        except rospy.ServiceException as e:
            rospy.logerr("Teleport failed: %s", e)

    def pose_callback(self, msg):
        self.pose = msg

    def follow_sine_path(self):
        while not rospy.is_shutdown():
            vel_msg = Twist()
            x = self.pose.x

            # Stop if turtle leaves the screen
            if x > 11.0:
                self.plot_path()
                break

            desired_y = self.amplitude * math.sin(self.frequency * x)
            dy_dx = self.amplitude * self.frequency * math.cos(self.frequency * x)
            desired_theta = math.atan2(dy_dx, 1.0)

            heading_error = desired_theta - self.pose.theta
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

            vel_msg.linear.x = self.linear_speed
            vel_msg.angular.z = self.k_heading * heading_error

            # Save actual and desired data for plotting
            self.actual_x.append(x)
            self.actual_y.append(self.pose.y)
            self.desired_y.append(desired_y)

            self.vel_pub.publish(vel_msg)
            self.rate.sleep()

    def plot_path(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.actual_x, self.desired_y, label='Desired Sine Curve', color='red', linestyle='--')
        plt.plot(self.actual_x, self.actual_y, label='Actual Turtle Path', color='blue')
        plt.title("Turtle Sine Wave Path Tracking")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    try:
        tracer = TurtleSineTracer()
        tracer.follow_sine_path()
    except rospy.ROSInterruptException:
        pass
