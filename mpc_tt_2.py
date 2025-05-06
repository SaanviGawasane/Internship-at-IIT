import rospy
import casadi as ca
import numpy as np
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import matplotlib.pyplot as plt

class TurtleMPC:
    def __init__(self):
        rospy.init_node('mpc_turtle_controller')
        self.pose = Pose()
        self.cmd_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)

        self.dt = 0.1
        self.N = 10  # Prediction horizon
        self.t = 0.0
        self.vel_log = []
        self.pose_log = []

        # Generate reference trajectory
        self.x_ref = np.linspace(1, 11, 1000)
        self.y_ref = 5 + 0.5 * np.sin(1.5 * self.x_ref)
        self.theta_ref = np.arctan(0.75 * np.cos(1.5 * self.x_ref))

        self.run()

    def pose_callback(self, msg):
        self.pose = msg

    def run(self):
        rate = rospy.Rate(1 / self.dt)

        while not rospy.is_shutdown():
            # Find the closest point in reference path
            idx = np.argmin(np.abs(self.x_ref - self.pose.x))
            if idx + self.N >= len(self.x_ref):
                break

            x_init = np.array([self.pose.x, self.pose.y, self.pose.theta])
            x_target = np.vstack([
                self.x_ref[idx+1:idx+1+self.N],
                self.y_ref[idx+1:idx+1+self.N],
                self.theta_ref[idx+1:idx+1+self.N]
            ])

            v, w = self.solve_mpc(x_init, x_target)
            self.publish_control(v, w)

            self.log_data(v, w)
            rate.sleep()

        self.plot_results()

    def solve_mpc(self, x0, ref):
        N = self.N
        dt = self.dt

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        n_states = states.size()[0]

        v = ca.SX.sym('v')
        w = ca.SX.sym('w')
        controls = ca.vertcat(v, w)
        n_controls = controls.size()[0]

        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), w)
        f = ca.Function('f', [states, controls], [rhs])

        X = ca.SX.sym('X', n_states, N+1)
        U = ca.SX.sym('U', n_controls, N)
        P = ca.SX.sym('P', n_states + N * n_states)

        obj = 0
        g = []

        Q = ca.diag([10, 10, 1])
        R = ca.diag([0.1, 0.1])

        st = X[:, 0]
        g.append(st - P[0:3])

        for k in range(N):
            st = X[:, k]
            con = U[:, k]
            st_next = X[:, k+1]
            f_value = f(st, con)
            st_next_euler = st + dt * f_value
            g.append(st_next - st_next_euler)

            ref_x = P[3 + k*3]
            ref_y = P[3 + k*3 + 1]
            ref_theta = P[3 + k*3 + 2]
            ref_state = ca.vertcat(ref_x, ref_y, ref_theta)
            obj += ca.mtimes([(st - ref_state).T, Q, (st - ref_state)])
            obj += ca.mtimes([con.T, R, con])

        OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp_prob = {'f': obj, 'x': OPT_variables, 'g': ca.vertcat(*g), 'p': P}

        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        lbg = [0] * (n_states * (N+1))
        ubg = [0] * (n_states * (N+1))

        x_init = np.zeros((n_states, N+1))
        u_init = np.zeros((n_controls, N))
        init_vals = np.concatenate([x_init.flatten(), u_init.flatten()])

        p = np.concatenate([x0, ref.T.flatten()])

        sol = solver(x0=init_vals, p=p, lbg=lbg, ubg=ubg)
        u = ca.reshape(sol['x'][n_states * (N+1):], n_controls, N)
        v_opt = float(u[0, 0])
        w_opt = float(u[1, 0])

        return v_opt, w_opt

    def publish_control(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

    def log_data(self, v, w):
        self.t += self.dt
        self.vel_log.append((self.t, v, w))
        self.pose_log.append((self.pose.x, self.pose.y))

    def plot_results(self):
        times, lin_vels, ang_vels = zip(*self.vel_log)
        xs, ys = zip(*self.pose_log)

        plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(xs, ys, label='Actual Path', color='blue')
        ax1.plot(self.x_ref, self.y_ref, '--', label='Desired Path', color='orange')
        ax1.set_title('Path')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(times, lin_vels, color='green')
        ax2.set_title('Linear Velocity vs Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Linear Velocity (m/s)')

        ax3 = plt.subplot(2, 2, 4)
        ax3.plot(times, ang_vels, color='red')
        ax3.set_title('Angular Velocity vs Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angular Velocity (rad/s)')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    try:
        TurtleMPC()
    except rospy.ROSInterruptException:
        pass
