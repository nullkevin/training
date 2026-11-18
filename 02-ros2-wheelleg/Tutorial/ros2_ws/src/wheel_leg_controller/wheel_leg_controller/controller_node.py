"""轮腿机器人控制器节点 —— 纯控制，不含 MuJoCo。

与仿真节点 wheel_leg_sim 只通过 ROS2 话题通信，各自独立 ros2 run 启动：

    订阅 /imu          sensor_msgs/msg/Imu         机体姿态四元数 + 角速度
    订阅 /joint_states sensor_msgs/msg/JointState  6 个关节位置/速度
    订阅 /cmd_vel      geometry_msgs/msg/Twist     目标速度 (linear.x) / 目标转向 (angular.z)
    发布 /actuator_cmd rm_msgs/msg/WheelLegCmd     6 个执行器力矩

时序：仿真节点是"指令驱动"的，每收到一条 /actuator_cmd 才走一个 dt=0.002 s。
因此本节点不设固定频率定时器，而是把每一条 /joint_states 当作控制周期时钟：
收到状态 → 解算 → 立刻发一条力矩。这样一条状态严格对应一步仿真。

控制解算复用 LQR_VMC 目录下的 vmc_cal / lqr_update / roll_leg_control /
ground_detection（已复制进本包），仅重写"ROS2 接口 + 控制回路编排"。
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rm_msgs.msg import WheelLegCmd
from sensor_msgs.msg import Imu, JointState

from .ground_detection import detect_leg_ground_status
from .lqr_update import lqr_controller
from .roll_leg_control import RollLegController
from .vmc_cal import five_links_param


class PIDController:
    """增量式离散 PID（沿用参考实现的参数定义）。"""

    def __init__(self, kp, ki, kd, max_out, max_i, feedforward=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.max_i = max_i
        self.feedforward = feedforward
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current, target):
        error = target - current
        self.integral = float(np.clip(self.integral + error, -self.max_i, self.max_i))
        derivative = error - self.prev_error
        self.prev_error = error
        output = (self.kp * error + self.ki * self.integral
                  + self.kd * derivative + self.feedforward)
        return float(np.clip(output, -self.max_out, self.max_out))


class WheelLegController(Node):

    def __init__(self):
        super().__init__('wheel_leg_controller')

        # ---------------- 参数 ----------------
        # mode:=openloop 固定/正弦；mode:=wheel_pi 轮速 PI；mode:=vmc 腿长 VMC；mode:=lqr 全闭环
        self.declare_parameter('mode', 'openloop')
        self.declare_parameter('test_rate', 200.0)           # 开环模式发布频率 (Hz)
        self.declare_parameter('test_amplitude', 0.3)        # 正弦力矩幅值 (N·m)
        self.declare_parameter('test_freq', 0.5)             # 正弦频率 (Hz，仿真时间)
        self.declare_parameter('wheel_radius', 0.06)
        self.declare_parameter('target_length', 0.10)        # 基础虚拟腿长 (m)
        self.declare_parameter('enable_ground_detection', False)
        self.declare_parameter('max_wheel_torque', 1.9)      # 执行器上限 ±2.0，留余量
        self.declare_parameter('accel_rate', 0.8)            # 速度斜坡 (m/s^2)
        self.declare_parameter('decel_rate', 1.4)
        self.declare_parameter('kp_wheel', 0.20)             # 轮速 PI：比例
        self.declare_parameter('ki_wheel', 0.80)             # 轮速 PI：积分
        self.declare_parameter('wheel_i_limit', 2.0)         # 积分限幅（抗饱和）
        self.declare_parameter('x_err_limit', 0.5)           # 位移误差限幅 (m)
        self.declare_parameter('x_err_breakaway', 0.9)       # 静止卡住时放宽的位移误差限幅 (m)
        self.declare_parameter('warmup_steps', 2500)         # 起步 settle 屏蔽步数（5 s 仿真时间）
        self.declare_parameter('tilt_limit', 0.12)           # 俯仰保护阈值 (rad)
        self.declare_parameter('v_cmd_limit', 0.3)           # 指令限幅 (m/s)，机械安全上限
        self.mode = str(self.get_parameter('mode').value).lower()
        self.test_rate = float(self.get_parameter('test_rate').value)
        self.test_amplitude = float(self.get_parameter('test_amplitude').value)
        self.test_freq = float(self.get_parameter('test_freq').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.target_length = float(self.get_parameter('target_length').value)
        self.enable_gd = bool(self.get_parameter('enable_ground_detection').value)
        self.max_wheel = float(self.get_parameter('max_wheel_torque').value)
        self.accel_rate = float(self.get_parameter('accel_rate').value)
        self.decel_rate = float(self.get_parameter('decel_rate').value)
        self.kp_wheel = float(self.get_parameter('kp_wheel').value)
        self.ki_wheel = float(self.get_parameter('ki_wheel').value)
        self.wheel_i_limit = float(self.get_parameter('wheel_i_limit').value)
        self.x_err_limit = float(self.get_parameter('x_err_limit').value)
        self.x_err_breakaway = float(self.get_parameter('x_err_breakaway').value)
        self.warmup_steps = int(self.get_parameter('warmup_steps').value)
        self.tilt_limit = float(self.get_parameter('tilt_limit').value)
        self.v_cmd_limit = float(self.get_parameter('v_cmd_limit').value)

        # 模型 timestep 固定 0.002 s，与仿真节点一致
        self.dt = 0.002

        # ---------------- 控制算法对象（左右腿各自独立状态） ----------------
        self.five_links_r = five_links_param(dt=self.dt)
        self.five_links_l = five_links_param(dt=self.dt)
        self.lqr_r = lqr_controller(dt=self.dt)
        self.lqr_l = lqr_controller(dt=self.dt)
        self.l0_pid_r = PIDController(700.0, 0.0, 5.0, 150.0, 0.0)
        self.l0_pid_l = PIDController(700.0, 0.0, 5.0, 150.0, 0.0)
        self.yaw_pid_r = PIDController(0.5, 0.0, 1.0, 2.0, 0.0)
        self.yaw_pid_l = PIDController(0.5, 0.0, 1.0, 2.0, 0.0)
        self.roll_balancer = RollLegController(
            kp=1.0, ki=0.0, kd=0.0, max_out=0.14, min_leg=0.10, max_leg=0.14)

        # ---------------- 运行时状态缓存 ----------------
        self.imu = None                  # 最近一帧 /imu
        self.pos = {}                    # 关节名 -> 位置
        self.vel = {}                    # 关节名 -> 速度
        self.target_v = 0.0              # 来自 /cmd_vel 的 linear.x
        self.target_w = 0.0              # 来自 /cmd_vel 的 angular.z
        self.current_v = 0.0             # 斜坡滤波后的 v_set
        self.robot_is_airborne = False
        self.steps = 0
        self.test_steps = 0
        self._rate_t0 = time.monotonic()
        self._rate_n0 = 0
        self._wheel_i = 0.0              # 轮速 PI 积分项
        self._x_ref = 0.0                # 位移参考（对速度指令积分）
        self._x_lim = self.x_err_limit   # 当前生效的位移误差限幅
        self._wheel_t0 = time.monotonic()

        # ---------------- 话题 ----------------
        self.cmd_pub = self.create_publisher(WheelLegCmd, 'actuator_cmd', 10)
        self.create_subscription(Imu, 'imu', self.on_imu, 10)
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)

        if self.mode == 'openloop':
            # 开环链路测试：定时发固定/正弦力矩，不读状态、不跑控制律
            period = 1.0 / max(self.test_rate, 1.0)
            self.create_timer(period, self.openloop_step)
            self.get_logger().info(
                f'开环测试模式 mode=openloop，{self.test_rate:.0f} Hz 正弦轮力矩 '
                f'幅值 {self.test_amplitude} N·m，频率 {self.test_freq} Hz（闭环请加 -p mode:=lqr）')
        elif self.mode == 'wheel_pi':
            self.get_logger().info(
                f'轮速 PI 模式 mode=wheel_pi：目标来自 /cmd_vel.linear.x，'
                f'kp={self.kp_wheel} ki={self.ki_wheel}（暂不做平衡）')
        elif self.mode == 'vmc':
            self.get_logger().info(
                f'VMC 腿长模式 mode=vmc：目标 L_0={self.target_length:.3f} m，'
                f'腿长 PID(700,0,5)，T_p=0，轮子不发力')
        else:
            self.get_logger().info(
                '闭环控制模式 mode=lqr：等待 /imu + /joint_states（每帧状态回一条 /actuator_cmd）')

    # ==================== 话题回调 ====================

    def on_imu(self, msg: Imu):
        self.imu = msg

    def on_cmd_vel(self, msg: Twist):
        # 限幅到机械安全速度：再快会迫使腿角顶到限位而失稳
        self.target_v = float(np.clip(msg.linear.x, -self.v_cmd_limit, self.v_cmd_limit))
        self.target_w = float(msg.angular.z)

    def on_joint_states(self, msg: JointState):
        """控制周期入口：以 /joint_states 为时钟，一帧状态对应一步仿真。"""
        self.pos = dict(zip(msg.name, msg.position))
        self.vel = dict(zip(msg.name, msg.velocity))
        if self.mode == 'wheel_pi':
            self.wheel_pi_step()        # 只闭环轮速，一帧状态回一条力矩
            return
        if self.mode == 'vmc':
            self.vmc_step()             # 只闭环虚拟腿长
            return
        if self.mode != 'lqr':
            return                      # 开环模式只缓存状态，由定时器发指令
        if self.imu is None:
            return                      # 还没收到 /imu，等下一帧（仿真有心跳会重发）
        try:
            self.control_step()
        except KeyError as exc:
            self.get_logger().error(f'关节数据缺失: {exc}', throttle_duration_sec=2.0)

    # ==================== 开环链路测试 ====================

    def openloop_step(self):
        """只发力矩、不读状态：腿关节固定 0，两轮同一正弦力矩。"""
        self.test_steps += 1
        t = self.test_steps * self.dt        # 用步数折算仿真时间，频率与墙钟无关
        wheel_t = self.test_amplitude * math.sin(2.0 * math.pi * self.test_freq * t)

        cmd = WheelLegCmd()
        cmd.right_phi1 = 0.0
        cmd.right_phi4 = 0.0
        cmd.left_phi1 = 0.0
        cmd.left_phi4 = 0.0
        cmd.right_wheel = wheel_t
        cmd.left_wheel = -wheel_t       # 左右轮安装镜像，取反才能同向滚动
        self.cmd_pub.publish(cmd)

        # 每 100 条打印一次实测发布频率 + 轮子反馈，直接验证"链路通、轮子有响应"
        if self.test_steps % 100 == 0:
            now = time.monotonic()
            rate = (self.test_steps - self._rate_n0) / max(now - self._rate_t0, 1e-6)
            self._rate_t0 = now
            self._rate_n0 = self.test_steps
            self.get_logger().info(
                f'已发 {self.test_steps} 条 /actuator_cmd | 实测 {rate:.1f} Hz '
                f'| 轮力矩 {wheel_t:+.3f} N·m '
                f'| 轮速 R={self.vel.get("rightwheel", 0.0):+.2f} '
                f'L={self.vel.get("leftwheel", 0.0):+.2f} rad/s')

    # ==================== 轮速 PI（阶段 2） ====================

    def wheel_pi_step(self):
        """只用 /joint_states 的轮速做 PI，跟随 /cmd_vel.linear.x（不做平衡）。

        左右轮镜像安装：同一前进方向下 rightwheel_vel 与 leftwheel_vel 符号相反，
        因此取 (right - left)/2 作为整车前进方向的等效轮速，且两轮的力矩要反号下发。
        """
        omega = (self.vel['rightwheel'] - self.vel['leftwheel']) / 2.0   # rad/s
        v_meas = omega * self.wheel_radius                               # m/s
        error = self.target_v - v_meas

        self._wheel_i = float(np.clip(self._wheel_i + error * self.dt,
                                     -self.wheel_i_limit, self.wheel_i_limit))
        torque = self.kp_wheel * error + self.ki_wheel * self._wheel_i
        torque = float(np.clip(torque, -self.max_wheel, self.max_wheel))

        cmd = WheelLegCmd()
        cmd.right_phi1 = 0.0
        cmd.right_phi4 = 0.0
        cmd.left_phi1 = 0.0
        cmd.left_phi4 = 0.0
        cmd.right_wheel = torque
        cmd.left_wheel = -torque
        self.cmd_pub.publish(cmd)

        self.steps += 1
        now = time.monotonic()
        if now - self._wheel_t0 >= 1.0:
            self._wheel_t0 = now
            err_pct = (abs(error) / abs(self.target_v) * 100.0
                       if abs(self.target_v) > 1e-3 else 0.0)
            self.get_logger().info(
                f'v_ref={self.target_v:+.3f} m/s | v_meas={v_meas:+.3f} m/s '
                f'| 误差 {error:+.3f} m/s ({err_pct:.1f}%) | 轮力矩 {torque:+.2f} N·m')

    # ==================== VMC 腿长（阶段 3） ====================

    def vmc_step(self):
        """用脚关节角度做五连杆正运动学，得到 L_0/theta，再用腿长 PID 出力矩。

        本阶段 T_p=0，只做"虚拟腿长"这一维；轮子不发力，也不做平衡。
        """
        if self.imu is not None:
            _, pitch = self.euler_from_quaternion(self.imu.orientation)
            pitch = -pitch
            gyro_pitch = -self.imu.angular_velocity.y
        else:
            pitch, gyro_pitch = 0.0, 0.0

        r1, r2, l0_r = self.leg_vmc(0, self.five_links_r, self.l0_pid_r,
                                    self.pos['right1'], self.pos['right2'],
                                    pitch, gyro_pitch)
        l1, l2, l0_l = self.leg_vmc(1, self.five_links_l, self.l0_pid_l,
                                    self.pos['left1'], self.pos['left2'],
                                    pitch, gyro_pitch)

        cmd = WheelLegCmd()
        cmd.right_phi1 = self.safe(r1)
        cmd.right_phi4 = self.safe(r2)
        cmd.left_phi1 = self.safe(l1)
        cmd.left_phi4 = self.safe(l2)
        cmd.right_wheel = 0.0
        cmd.left_wheel = 0.0
        self.cmd_pub.publish(cmd)

        self.steps += 1
        now = time.monotonic()
        if now - self._wheel_t0 >= 1.0:
            self._wheel_t0 = now
            self.get_logger().info(
                f'L_0 目标 {self.target_length * 1000:.1f} mm | '
                f'右 {l0_r * 1000:.1f} mm ({(l0_r - self.target_length) * 1000:+.1f}) | '
                f'左 {l0_l * 1000:.1f} mm ({(l0_l - self.target_length) * 1000:+.1f})')

    def leg_vmc(self, leg_flag, five_links, l0_pid, m1, m2, pitch, gyro_pitch):
        """单腿：正运动学 -> 腿长 PID -> 虚拟轴向力 -> VMC 雅可比映射成两个髋关节力矩。"""
        five_links.forward_kinematics_cal(leg_flag, m1, m2, pitch, gyro_pitch)

        f0 = l0_pid.compute(five_links.L_0, self.target_length)
        cos_theta = math.cos(five_links.theta)
        f0 += 4.0 / cos_theta if abs(cos_theta) > 1e-6 else 0.0
        f0 = float(np.clip(f0, -100.0, 100.0))

        t1, t2 = five_links.VMC_torque_cal(f0, 0.0)     # T_p = 0，只保留轴向力
        sign = 1.0 if leg_flag == 0 else -1.0          # 左腿电机镜像
        return sign * t1, sign * t2, five_links.L_0

    # ==================== 控制回路 ====================

    def control_step(self):
        # 1. 四元数 -> 欧拉角（与参考实现的坐标系/公式完全一致）
        roll, pitch = self.euler_from_quaternion(self.imu.orientation)
        gx = self.imu.angular_velocity.x
        gy = self.imu.angular_velocity.y
        # 参考实现的符号约定
        pitch = -pitch
        roll = -roll
        gyro_pitch = -gy
        gyro_yaw = gx

        # 2. 速度指令斜坡，避免阶跃激励
        self.current_v = self.ramp(self.current_v, self.target_v)

        # 3. 横滚平衡：把基础腿长差模分配给左右腿
        target_len_l, target_len_r = self.roll_balancer.compute_leg_lengths(
            current_roll=roll, target_roll=0.0, base_length=self.target_length)

        # 4. 共模提取（两轮位置/速度的平均）
        common_pos = (self.pos['leftwheel'] - self.pos['rightwheel']) / 2.0
        common_speed = (self.vel['leftwheel'] - self.vel['rightwheel']) / 2.0
        # 物理 +x 速度（正 = 机体沿 +x 前进），仅用于指令/日志
        v_meas = common_speed * self.wheel_radius

        # 5. 起步保护：仿真启动后机体有一段下落/settle 瞬态，期间只做平衡，
        #    不响应行驶指令；俯仰过大时同样冻结位移参考，避免"带病行驶"。
        #    另有指令但轮子几乎不转（腿角顶到限位卡住）时放宽位移误差限幅，
        #    用更大的推力把机体"拔"出死区。
        self._x_lim = self.x_err_limit
        if abs(v_meas) < 0.05 and abs(self.current_v) > 0.1:
            self._x_lim = self.x_err_breakaway
        if self.steps < self.warmup_steps:
            self.target_v = 0.0
            self.current_v = 0.0
        elif abs(pitch) > self.tilt_limit:
            self._x_ref = common_pos * self.wheel_radius

        if self.enable_gd and self.robot_is_airborne:
            self._x_ref = common_pos * self.wheel_radius
            self.current_v = 0.0
            self.target_w = 0.0
        else:
            self._x_ref += self.current_v * self.dt

        self.lqr_l.x_b_set = self._x_ref
        self.lqr_r.x_b_set = self._x_ref
        self.lqr_l.v_set = self.current_v
        self.lqr_r.v_set = self.current_v

        # 6. 双腿独立解算（右腿 leg_flag=0，左腿 leg_flag=1）
        r1, r2, rw, flight_r = self.leg_control(
            0, self.five_links_r, self.lqr_r, self.yaw_pid_r, self.l0_pid_r,
            pitch, gyro_pitch, gyro_yaw, target_len_r,
            common_pos, common_speed, self.robot_is_airborne)
        l1, l2, lw, flight_l = self.leg_control(
            1, self.five_links_l, self.lqr_l, self.yaw_pid_l, self.l0_pid_l,
            pitch, gyro_pitch, gyro_yaw, target_len_l,
            common_pos, common_speed, self.robot_is_airborne)

        if self.enable_gd:
            self.robot_is_airborne = (flight_r == 1) or (flight_l == 1)

        # 7. 下发
        cmd = WheelLegCmd()
        cmd.right_phi1 = self.safe(r1)
        cmd.right_phi4 = self.safe(r2)
        cmd.left_phi1 = self.safe(l1)
        cmd.left_phi4 = self.safe(l2)
        cmd.right_wheel = self.safe(rw)
        cmd.left_wheel = self.safe(lw)
        self.cmd_pub.publish(cmd)

        self.steps += 1
        if self.steps % 1000 == 0:
            x_err = float(np.clip(self.lqr_r.x_b - self.lqr_r.x_b_set,
                                  -self._x_lim, self._x_lim))
            k0 = self.lqr_r.K[0] * self.lqr_r.theta
            k4 = self.lqr_r.K[4] * x_err
            k6 = self.lqr_r.K[6] * (self.lqr_r.v_b_whole - self.lqr_r.v_set)
            k8 = self.lqr_r.K[8] * self.lqr_r.phi
            self.get_logger().info(
                f'步数 {self.steps} | pitch={pitch:+.3f} rad | v_set={self.current_v:+.2f} m/s '
                f'| v_meas={v_meas:+.3f} m/s | x_err={x_err:+.3f} m '
                f'| 腿长 {self.five_links_r.L_0:.3f}/{self.five_links_l.L_0:.3f} m '
                f'| theta={self.lqr_r.theta:+.3f} phi={self.lqr_r.phi:+.3f} '
                f'| k0={k0:+.2f} k4={k4:+.2f} k6={k6:+.2f} k8={k8:+.2f} '
                f'| T=({r1:+.2f},{r2:+.2f},{l1:+.2f},{l2:+.2f},{rw:+.2f},{lw:+.2f})')

    def leg_control(self, leg_flag, five_links, lqr, yaw_pid, l0_pid,
                    pitch, gyro_pitch, gyro_yaw, target_length,
                    common_pos, common_speed, airborne_last):
        """单腿：VMC 正运动学 -> LQR 状态更新 -> 虚拟力/力矩 -> 关节力矩映射。"""
        if leg_flag == 0:               # 右腿
            m1, m2 = self.pos['right1'], self.pos['right2']
            # 左/右轮安装方向相反，用共模量时需取反，保证两腿状态定义一致
            wheel_pos, wheel_vel = -common_pos, -common_speed
            sign = 1.0
        else:                           # 左腿
            m1, m2 = self.pos['left1'], self.pos['left2']
            wheel_pos, wheel_vel = common_pos, common_speed
            sign = -1.0

        # 1. VMC 五连杆正运动学 + LQR 状态/时变 K
        five_links.forward_kinematics_cal(leg_flag, m1, m2, pitch, gyro_pitch)
        lqr.states_update(leg_flag, five_links, pitch, gyro_pitch,
                          wheel_pos, wheel_vel, self.wheel_radius)
        lqr.calc_k_matrix_from_poly(five_links.L_0)

        # 2. 腿长闭环 + 重力前馈 -> 虚拟轴向力 F_0
        f0_ctrl = l0_pid.compute(five_links.L_0, target_length)
        cos_theta = math.cos(five_links.theta)
        f0 = (4.0 / cos_theta if abs(cos_theta) > 1e-6 else 0.0) + f0_ctrl
        f0 = float(np.clip(f0, -100.0, 100.0))

        # 3. LQR -> 虚拟劈叉力矩 T_p（位移误差做限幅，防止长时间落后把腿顶死）
        x_err = float(np.clip(lqr.x_b - lqr.x_b_set, -self._x_lim, self._x_lim))
        t_p = (lqr.K[1] * lqr.theta
               + lqr.K[3] * lqr.d_theta
               + lqr.K[5] * x_err
               + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
               + lqr.K[9] * lqr.phi
               + lqr.K[11] * lqr.d_phi)

        if self.enable_gd and (detect_leg_ground_status(five_links, f0, t_p) == 1
                               or airborne_last):
            # 离地：轮子不发力，只保留髋关节姿态阻尼
            wheel_t = 0.0
            t_p = lqr.K[1] * lqr.theta + lqr.K[3] * lqr.d_theta
            flight = 1
        else:
            yaw_ctrl = yaw_pid.compute(gyro_yaw, self.target_w)
            wheel_t = sign * (lqr.K[0] * lqr.theta
                              + lqr.K[2] * lqr.d_theta
                              + lqr.K[4] * x_err
                              + lqr.K[6] * (lqr.v_b_whole - lqr.v_set)
                              + lqr.K[8] * lqr.phi
                              + lqr.K[10] * lqr.d_phi) - yaw_ctrl
            wheel_t = float(np.clip(wheel_t, -self.max_wheel, self.max_wheel))
            flight = 0

        # 4. VMC 雅可比把 [F_0, T_p] 映射成两个髋关节力矩
        t1, t2 = five_links.VMC_torque_cal(f0, t_p)
        return sign * t1, sign * t2, wheel_t, flight

    # ==================== 工具函数 ====================

    @staticmethod
    def euler_from_quaternion(q):
        """ROS 四元数 (x,y,z,w) -> (roll, pitch)，公式与参考实现一致。"""
        w, x, y, z = q.w, q.x, q.y, q.z
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = max(-1.0, min(1.0, t2))    # 防止 asin 出界
        pitch = math.asin(t2)
        return roll, pitch

    def ramp(self, current, target):
        """速度斜坡：按加速度/减速度限幅逼近目标值。"""
        delta = target - current
        rate = self.accel_rate if abs(target) > abs(current) else self.decel_rate
        step = rate * self.dt
        if abs(delta) <= step:
            return target
        return current + math.copysign(step, delta)

    @staticmethod
    def safe(value):
        """非有限值一律按 0 下发，避免把 NaN 打进执行器。"""
        return float(value) if math.isfinite(value) else 0.0


def main(args=None):
    rclpy.init(args=args)
    node = WheelLegController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
