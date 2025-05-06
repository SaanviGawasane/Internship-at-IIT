#!/usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import TwistStamped
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry

class IrisAutoTakeoffHover:
    def __init__(self):
        rospy.init_node("iris_auto_takeoff_hover")
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback)
        self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback)
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", TwistStamped, queue_size=10)

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")

        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        self.current_state = State()
        self.current_altitude = 0.0
        self.current_position = [0.0, 0.0, 0.0]
        self.path_x, self.path_y = [], []

        self.rate = rospy.Rate(20)

    def state_callback(self, msg):
        self.current_state = msg

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_altitude = pos.z
        self.current_position = [pos.x, pos.y, pos.z]
        self.path_x.append(pos.x)
        self.path_y.append(pos.y)

    def wait_for_connection(self):
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        rospy.loginfo("Connected to FCU.")

    def send_initial_setpoints(self, duration=5):
        rospy.loginfo("Sending initial setpoints...")
        twist = TwistStamped()
        for _ in range(int(duration * 20)):
            twist.header.stamp = rospy.Time.now()
            self.vel_pub.publish(twist)
            self.rate.sleep()

    def set_offboard_and_arm(self):
        rospy.loginfo("Switching to OFFBOARD mode...")
        result = self.set_mode_client(custom_mode="OFFBOARD")
        if result.mode_sent:
            rospy.loginfo("OFFBOARD mode set.")
        else:
            rospy.logwarn("Failed to set OFFBOARD mode.")
        rospy.sleep(0.5)

        rospy.loginfo("Arming drone...")
        arm_result = self.arming_client(True)
        if arm_result.success:
            rospy.loginfo("Drone armed.")
        else:
            rospy.logwarn("Arming failed.")
        rospy.sleep(0.5)

    def takeoff_to_altitude(self, target_alt=2.5, vz=0.6):
        rospy.loginfo(f"Taking off to {target_alt} m...")
        twist = TwistStamped()
        twist.twist.linear.z = vz
        while not rospy.is_shutdown():
            if self.current_altitude >= target_alt:
                rospy.loginfo(f"Reached altitude: {self.current_altitude:.2f} m")
                break
            twist.header.stamp = rospy.Time.now()
            self.vel_pub.publish(twist)
            self.rate.sleep()

        # Hover
        hover_twist = TwistStamped()
        hover_twist.twist.linear.z = 0.0
        rospy.loginfo("Hovering at target altitude...")
        for _ in range(40):
            hover_twist.header.stamp = rospy.Time.now()
            self.vel_pub.publish(hover_twist)
            self.rate.sleep()

    def wait_for_velocity_input_loop(self):
        rospy.loginfo("Ready for velocity input (X, Y). Type Ctrl+C to stop.")
        try:
            while not rospy.is_shutdown():
                lin_x = float(input("Enter linear X velocity: "))
                lin_y = float(input("Enter linear Y velocity: "))
                vel = TwistStamped()
                vel.twist.linear.x = lin_x
                vel.twist.linear.y = lin_y
                vel.twist.linear.z = 0.0
                vel.twist.angular.z = 0.0
                rospy.loginfo(f"Publishing X={lin_x}, Y={lin_y}")
                for _ in range(20):
                    vel.header.stamp = rospy.Time.now()
                    self.vel_pub.publish(vel)
                    self.rate.sleep()
        except (KeyboardInterrupt, ValueError):
            rospy.loginfo("Manual input stopped.")

    def return_to_start(self, return_speed=0.5):
        rospy.loginfo("Returning to start...")
        start_x, start_y = self.path_x[0], self.path_y[0]

        while not rospy.is_shutdown():
            cur_x, cur_y = self.current_position[0], self.current_position[1]
            dx, dy = start_x - cur_x, start_y - cur_y
            dist = (dx**2 + dy**2)**0.5
            if dist < 0.2:
                rospy.loginfo("Reached starting point.")
                break

            norm = max(dist, 1e-6)
            vx = return_speed * dx / norm
            vy = return_speed * dy / norm

            vel = TwistStamped()
            vel.twist.linear.x = vx
            vel.twist.linear.y = vy
            vel.twist.linear.z = 0.0
            vel.header.stamp = rospy.Time.now()
            self.vel_pub.publish(vel)
            self.rate.sleep()

        # Hover briefly before stopping
        stop = TwistStamped()
        stop.twist.linear.z = 0.0
        for _ in range(40):
            stop.header.stamp = rospy.Time.now()
            self.vel_pub.publish(stop)
            self.rate.sleep()

    def plot_trajectory(self):
        plt.figure()
        plt.plot(self.path_x, self.path_y, label='Drone path')
        plt.scatter(self.path_x[0], self.path_y[0], c='green', label='Start')
        plt.scatter(self.path_x[-1], self.path_y[-1], c='red', label='End')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Drone Trajectory')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()

    def run(self):
        self.wait_for_connection()
        self.send_initial_setpoints()
        self.set_offboard_and_arm()
        rospy.sleep(1)
        self.takeoff_to_altitude()
        self.wait_for_velocity_input_loop()
        self.return_to_start()
        self.plot_trajectory()

if __name__ == "__main__":
    try:
        IrisAutoTakeoffHover().run()
    except rospy.ROSInterruptException:
        pass
