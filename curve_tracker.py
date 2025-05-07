#!/usr/bin/env python3
import rospy
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import TwistStamped
from mavros_msgs.srv   import CommandBool, SetMode
from mavros_msgs.msg   import State
from nav_msgs.msg      import Odometry
import tf

class CasADiMPCSineTracker:
    def __init__(self):
        rospy.init_node("casadi_mpc_sine_tracker")
        self.rate = rospy.Rate(10)  # 10 Hz loop

        # MAVROS state & pose
        self.state = None
        self.pose  = None
        self.yaw   = 0.0

        # Logs
        self.logs = {k: [] for k in
            ("t","vx","vy","z","yaw","x","y","xd","yd")}

        # ROS interfaces
        rospy.Subscriber("/mavros/state", State,    self._state_cb)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self._odom_cb)
        self._vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel",
                                        TwistStamped, queue_size=1)
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        self._arm_srv  = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self._mode_srv = rospy.ServiceProxy("/mavros/set_mode",   SetMode)

        # Build MPC
        self._build_mpc(horizon=10, dt=0.1)

        # Startup
        self._wait_for_fcu()
        self._warmup_setpoints()
        self._set_offboard_and_arm()
        self._takeoff(2.0)

        # Record start point
        self.start_x = self.pose.position.x
        self.start_y = self.pose.position.y

        # Run MPC for 30 s
        self._run(duration=30.0, dt=0.1)

        # Stop and plot
        self._vel_pub.publish(TwistStamped())
        rospy.sleep(1.0)
        self._plot()

    #— callbacks
    def _state_cb(self, msg):   self.state = msg
    def _odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = p
        q = p.orientation
        _,_,self.yaw = tf.transformations.euler_from_quaternion(
            [q.x,q.y,q.z,q.w])

    #— startup helpers
    def _wait_for_fcu(self):
        rospy.loginfo("Waiting for FCU...")
        while not rospy.is_shutdown() and (not self.state or not self.state.connected):
            self.rate.sleep()

    def _warmup_setpoints(self):
        rospy.loginfo("Warming up setpoints...")
        dummy = TwistStamped()
        for _ in range(100):
            dummy.header.stamp = rospy.Time.now()
            self._vel_pub.publish(dummy)
            self.rate.sleep()

    def _set_offboard_and_arm(self):
        rospy.loginfo("Setting OFFBOARD...")
        if not self._mode_srv(0, "OFFBOARD").mode_sent:
            rospy.logerr("OFFBOARD failed"); exit(1)
        rospy.loginfo("Arming...")
        if not self._arm_srv(True).success:
            rospy.logerr("Arming failed"); exit(1)

    def _takeoff(self, z_target):
        rospy.loginfo(f"Takeoff to {z_target} m...")
        cmd = TwistStamped()
        cmd.twist.linear.z = 0.5
        while not rospy.is_shutdown():
            if self.pose and self.pose.position.z >= z_target - 0.1:
                break
            cmd.header.stamp = rospy.Time.now()
            self._vel_pub.publish(cmd)
            self.rate.sleep()
        rospy.loginfo("Reached takeoff height.")

    #— MPC builder
    def _build_mpc(self, horizon, dt):
        N = horizon
        # weights
        Q = np.diag([10, 10, 1])     # x, y, yaw
        R = np.diag([1.0, 1.0, 0.1])  # vx, vy, w

        opti = ca.Opti()
        X = opti.variable(3, N+1)   # states: x,y,yaw
        U = opti.variable(3, N)     # controls: vx,vy,w
        P = opti.parameter(3 + 3*N) # x0,y0,yaw0 + N×(x_ref,y_ref,yaw_ref)

        obj = 0
        for k in range(N):
            st  = X[:,k]
            ref = P[3+3*k:3+3*k+3]
            u   = U[:,k]
            # cost
            obj += ca.mtimes([(st-ref).T, Q, (st-ref)]) \
                 + ca.mtimes([u.T,      R,      u])
            # dynamics
            st_next = X[:,k+1]
            f = ca.vertcat(u[0], u[1], u[2])
            opti.subject_to(st_next == st + dt*f)

            # bounds per step
            opti.subject_to(opti.bounded(-2.0, U[0,k], 2.0))
            opti.subject_to(opti.bounded(-2.0, U[1,k], 2.0))
            opti.subject_to(opti.bounded(-1.0, U[2,k], 1.0))

        # initial state constraint
        opti.subject_to(X[:,0] == P[0:3])

        opti.minimize(obj)
        opti.solver("ipopt", {"print_time":False}, {"print_level":0})

        # store references
        self.opti    = opti
        self.X_var   = X
        self.U_var   = U
        self.P_par   = P
        self.N       = N
        self.dt      = dt

        # warm start
        opti.set_initial(U, 0)

    #— desired trajectory
    def _desired(self, t):
        speed_sp = 1.0
        A        = 2.0
        ω        = 0.5
        x_d  = self.start_x + speed_sp*t
        y_d  = self.start_y + A*np.sin(ω*t)
        yaw_d= np.arctan2(A*ω*np.cos(ω*t), speed_sp)
        return x_d, y_d, yaw_d

    #— main loop
    def _run(self, duration, dt):
        steps = int(duration/dt)
        for i in range(steps):
            t = i*dt
            if not self.pose:
                self.rate.sleep()
                continue

            # current
            x0, y0 = self.pose.position.x, self.pose.position.y
            ψ0      = self.yaw

            # build parameters: [x0,y0,ψ0, ref1, ref2, … refN]
            param = [x0, y0, ψ0]
            for k in range(self.N):
                tk = t + (k+1)*dt
                param += list(self._desired(tk))

            self.opti.set_value(self.P_par, np.array(param))

            sol = self.opti.solve()
            vx, vy, w = sol.value(self.U_var[:,0])

            # publish
            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.twist.linear.x  = vx
            cmd.twist.linear.y  = vy
            cmd.twist.linear.z  = 0
            cmd.twist.angular.z = w
            self._vel_pub.publish(cmd)

            # log
            L = self.logs
            L["t"].append(t)
            L["vx"].append(vx)
            L["vy"].append(vy)
            L["z"].append(self.pose.position.z)
            L["yaw"].append(self.yaw)
            L["x"].append(x0)
            L["y"].append(y0)
            xd, yd, _ = self._desired(t)
            L["xd"].append(xd)
            L["yd"].append(yd)

            self.rate.sleep()

    #— plotting
    def _plot(self):
        L = self.logs
        # XY path
        plt.figure(figsize=(6,6))
        plt.plot(L["xd"], L["yd"], 'r--', label="Desired")
        plt.plot(L["x"],  L["y"],  'b-',  label="Actual")
        plt.title("XY Trajectory"); plt.xlabel("X"); plt.ylabel("Y")
        plt.legend(); plt.grid(); plt.axis('equal')
        plt.show()

        # vx, vy, z, yaw vs t
        fig, axs = plt.subplots(2,2, figsize=(10,8))
        axs[0,0].plot(L["t"],L["vx"]); axs[0,0].set_title("Vx vs Time")
        axs[0,1].plot(L["t"],L["vy"]); axs[0,1].set_title("Vy vs Time")
        axs[1,0].plot(L["t"],L["z"]);  axs[1,0].set_title("Z vs Time")
        axs[1,1].plot(L["t"],L["yaw"]);axs[1,1].set_title("Yaw vs Time")
        for ax in axs.flat:
            ax.set_xlabel("Time (s)"); ax.grid(True)
        plt.tight_layout(); plt.show()

if __name__=="__main__":
    try:
        CasADiMPCSineTracker()
    except rospy.ROSInterruptException:
        pass
