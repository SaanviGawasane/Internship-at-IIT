#!/usr/bin/env python3

import rospy
import math
import time
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from std_srvs.srv import Empty
from turtlesim.srv import TeleportAbsolute

class TurtleSineTracer:
    def __init__(self):
        rospy.init_node('turtle_sine_tracer', anonymous=True)

        self.vel_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)

        self.pose = Pose()
        self.latest_twist = Twist()
        self.rate = rospy.Rate(20)

        # Sine wave parameters
        self.amplitude = 2.0
        self.frequency = 1.5
        self.angular_speed = 3.0
        self.linear_speed = 1.0

        self.actual_x = []
        self.actual_y = []
        self.desired_y = []

        self.run_id = int(time.time())

        # Teleport to start
        self.teleport_to_start()

        # Wait for pose
        rospy.loginfo("Waiting for initial pose...")
        while self.pose.x == 0.0 and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo(f"Starting from pose: x = {self.pose.x:.2f}, y = {self.pose.y:.2f}")

    def pose_callback(self, msg):
        self.pose = msg

    def teleport_to_start(self):
        rospy.wait_for_service('/turtle1/teleport_absolute')
        try:
            teleport = rospy.ServiceProxy('/turtle1/teleport_absolute', TeleportAbsolute)
            teleport(1.0, 5.5, 0.0)
            rospy.loginfo("Teleported turtle to (1, 5.5)")
        except rospy.ServiceException as e:
            rospy.logerr("Teleport service failed: %s" % e)

    def follow_sine_for_5_seconds(self):
        start_time = time.time()

        while not rospy.is_shutdown():
            elapsed = time.time() - start_time
            if elapsed > 5.0:
                break

            vel_msg = Twist()
            x = self.pose.x

            vertical_offset = 5.5 - self.amplitude * math.sin(self.frequency * 1.0)
            desired_y = self.amplitude * math.sin(self.frequency * x) + vertical_offset
            dy_dx = self.amplitude * self.frequency * math.cos(self.frequency * x)
            desired_theta = math.atan2(dy_dx, 1.0)

            heading_error = desired_theta - self.pose.theta
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

            vel_msg.linear.x = self.linear_speed
            vel_msg.angular.z = self.angular_speed * heading_error

            self.latest_twist = vel_msg
            self.actual_x.append(x)
            self.actual_y.append(self.pose.y)
            self.desired_y.append(desired_y)

            self.vel_pub.publish(vel_msg)
            self.rate.sleep()

        self.vel_pub.publish(Twist())
        rospy.sleep(0.5)

        self.save_data()
        self.plot_run()

        print("\n After 5 seconds:")
        print(f" X Position      : {self.pose.x:.2f}")
        print(f" Y Position      : {self.pose.y:.2f}")
        print(f" Linear Velocity : {self.latest_twist.linear.x:.2f}")
        print(f" Angular Velocity: {self.latest_twist.angular.z:.2f}")

    def save_data(self):
        filename = "turtle_path_data.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["run_id", "x", "actual_y", "desired_y"])
            for i in range(len(self.actual_x)):
                writer.writerow([self.run_id, self.actual_x[i], self.actual_y[i], self.desired_y[i]])
        #print(f" Path data saved to {filename}")

    def plot_run(self):
        df = pd.read_csv("turtle_path_data.csv")
        runs = df['run_id'].unique()

        plt.figure(figsize=(10, 6))
        for run_id in runs:
            run_data = df[df['run_id'] == run_id]
            plt.plot(run_data['x'], run_data['actual_y'], label=f"Run {run_id} - Actual", linewidth=2)
            plt.plot(run_data['x'], run_data['desired_y'], '--', color='gray', label=f"Run {run_id} - Desired")

        plt.title("Turtle Path vs Desired Sine Curve (Auto-Generated)")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("turtle_path_plot.png")
        plt.show()
        #print(" Plot displayed and saved as turtle_path_plot.png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', action='store_true', help='Clear previous CSV data before running')
    args = parser.parse_args()

    if args.clear:
        if os.path.exists("turtle_path_data.csv"):
            os.remove("turtle_path_data.csv")
            #print(" Previous path data cleared.")

    try:
        tracer = TurtleSineTracer()
        tracer.follow_sine_for_5_seconds()
    except rospy.ROSInterruptException:
        pass

