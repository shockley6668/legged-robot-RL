import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, Vector3
import os

from .rdk_inference import TinkerRealInference

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # Parameters
        self.declare_parameter('model_path', '/root/legged-robot/src/robot_control/robot_control/model_1500.onnx')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        print("model_path:", model_path)
        self.get_logger().info(f'Loading model from: {model_path}')
        
        try:
            self.inference = TinkerRealInference(model_path)
            self.get_logger().info('Model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            # We might want to exit or handle this, but for now we'll let it crash later if used
            self.inference = None

        # State variables
        self.latest_euler = np.zeros(3, dtype=np.float32) # roll, pitch, yaw
        self.latest_imu_gyro = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # 10 joints
        self.latest_joint_pos = np.zeros(10, dtype=np.float32)
        self.latest_joint_vel = np.zeros(10, dtype=np.float32)
        
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_dyaw = 0.0

        # Subscriptions
        self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        self.create_subscription(Float64MultiArray, 'imu/data', self.imu_callback, 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)

        # Publisher
        self.motor_cmd_pub = self.create_publisher(Float64MultiArray, 'motor_cmds', 10)

        # Timer for inference loop (50Hz)
        self.dt = 0.02
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info('Inference node started')

    def joint_callback(self, msg: JointState):
        # Taking the first 10 joints assuming they match the 10 motors in order
        # bridge_node publishes joint_0 to joint_9 in order.
        if len(msg.position) >= 10:
            self.latest_joint_pos = np.array(msg.position[:10], dtype=np.float32)
        if len(msg.velocity) >= 10:
            self.latest_joint_vel = np.array(msg.velocity[:10], dtype=np.float32)

    def imu_callback(self, msg: Float64MultiArray):
        # Format: [roll, pitch, yaw, gyro_x, gyro_y, gyro_z]
        if len(msg.data) >= 6:
            self.latest_euler = np.array(msg.data[0:3], dtype=np.float32)
            self.latest_imu_gyro = np.array(msg.data[3:6], dtype=np.float32)

    def cmd_callback(self, msg: Twist):
        self.cmd_vx = msg.linear.x
        self.cmd_vy = msg.linear.y
        self.cmd_dyaw = msg.angular.z

    def timer_callback(self):
        if self.inference is None:
            return

        # Run inference
        try:
            target_q = self.inference.get_action(
                self.latest_euler,
                self.latest_imu_gyro,
                self.latest_joint_pos,
                self.latest_joint_vel,
                self.cmd_vx,
                self.cmd_vy,
                self.cmd_dyaw
            )

            # Publish result
            cmd_msg = Float64MultiArray()
            cmd_msg.data = target_q.tolist() # Convert numpy array to list
            # print(target_q)
            self.motor_cmd_pub.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error during inference: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
