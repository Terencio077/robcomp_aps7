import time

from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data


class Odom:
    def __init__(self):
        self.x = None
        self.y = None
        self.ultima_odom = 0.0
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.ultima_odom = time.monotonic()
