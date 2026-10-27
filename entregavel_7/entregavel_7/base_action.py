import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


class Acao(Node):
    def __init__(self, nome):
        Node.__init__(self, nome)
        self.timer = None
        self.robot_state = 'done'
        self.twist = Twist()
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_machine = {'stop': self.stop, 'done': self.done}

    def reset(self, estado):
        self.twist = Twist()
        self.robot_state = estado
        if self.timer is None:
            self.timer = self.create_timer(0.05, self.control)

    def stop(self):
        self.twist = Twist()
        if rclpy.ok():
            self.cmd_vel_pub.publish(self.twist)
        if self.timer is not None:
            self.destroy_timer(self.timer)
            self.timer = None
        self.robot_state = 'done'

    def done(self):
        self.twist = Twist()

    def control(self):
        self.twist = Twist()
        self.state_machine[self.robot_state]()
        self.cmd_vel_pub.publish(self.twist)


def executar(classe, args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = classe()
        node.reset()
        while rclpy.ok() and node.robot_state != 'done':
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
