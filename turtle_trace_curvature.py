#!/usr/bin/env python3

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
import math
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import signal
import sys

class TurtleCurvatureFollower:
    def __init__(self):
        rospy.init_node('turtle_live_plot_curvature', anonymous=True)
        self.pose = None

        rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)
        self.pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)

        # Parameters
        self.amplitude = 2.0
        self.frequency = 1.0
        self.linear_velocity = 1.0
        self.dt = 0.1
        self.csv_file = "turtle_curvature_data.csv"

        # Setup live plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line_actual, = self.ax.plot([], [], 'bo-', label="Actual Path")
        self.line_desired, = self.ax.plot([], [], 'r--', label="Desired Sine Curve")
        self.ax.set_title("Live Turtle Path vs Desired Sine Curve")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)
        self.ax.legend()
        self.x_data, self.y_actual_data, self.y_desired_data = [], [], []

        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)

        rospy.wait_for_service('/turtle1/teleport_absolute')
        try:
            from turtlesim.srv import TeleportAbsolute
            teleport = rospy.ServiceProxy('/turtle1/teleport_absolute', TeleportAbsolute)
            teleport(1.0, 5.5, 0.0)  # x=1, y=5.5, theta=0
        except rospy.ServiceException as e:
            rospy.logerr("Teleport failed: %s", e)

        rospy.wait_for_message('/turtle1/pose', Pose)

        # Shutdown hook to show final plot
        signal.signal(signal.SIGINT, self.signal_handler)

    def pose_callback(self, msg):
        self.pose = msg

    def compute_radius_of_curvature(self, x):
        A = self.amplitude
        f = self.frequency

        dy_dx = A * f * math.cos(f * x)
        d2y_dx2 = -A * f**2 * math.sin(f * x)

        numerator = (1 + dy_dx**2)**1.5
        denominator = abs(d2y_dx2) if d2y_dx2 != 0 else 1e-5

        R = numerator / denominator
        return R, dy_dx

    def update_live_plot(self):
        self.line_actual.set_data(self.x_data, self.y_actual_data)
        self.line_desired.set_data(self.x_data, self.y_desired_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        rate = rospy.Rate(1 / self.dt)

        while not rospy.is_shutdown():
            if self.pose is None:
                continue

            x = self.pose.x
            y = self.pose.y
            desired_y = self.amplitude * math.sin(self.frequency * x)
            R, dy_dx = self.compute_radius_of_curvature(x)
            omega = self.linear_velocity / R

            desired_theta = math.atan(dy_dx)
            heading_error = desired_theta - self.pose.theta
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            corrected_omega = heading_error / self.dt

            # Publish command
            twist = Twist()
            twist.linear.x = self.linear_velocity
            twist.angular.z = corrected_omega
            self.pub.publish(twist)

            # Save data
            self.x_data.append(x)
            self.y_actual_data.append(y)
            self.y_desired_data.append(desired_y)

            file_exists = os.path.isfile(self.csv_file)
            with open(self.csv_file, 'a') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["x", "y_actual", "y_desired", "linear_x", "angular_z", "R"])
                writer.writerow([x, y, desired_y, self.linear_velocity, corrected_omega, R])

            # Update live plot
            self.update_live_plot()

            rate.sleep()

    def signal_handler(self, sig, frame):
        #print("\n Stopped. Showing final plot...")
        plt.ioff()
        plt.show()
        sys.exit(0)

if __name__ == "__main__":
    try:
        follower = TurtleCurvatureFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
