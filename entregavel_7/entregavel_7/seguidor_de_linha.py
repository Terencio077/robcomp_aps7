import math
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from .odom import Odom
from .segue_linha import SegueLinha


class SeguidorDeLinha(Node, Odom):
    def __init__(self):
        Node.__init__(self, 'seguidor_de_linha_node')
        Odom.__init__(self)
        self.declare_parameter('distancia_saida', 0.8)
        self.declare_parameter('distancia_chegada', 0.25)
        self.declare_parameter('raio_chegada', 0.6)
        self.distancia_saida = self.get_parameter('distancia_saida').value
        self.distancia_chegada = self.get_parameter('distancia_chegada').value
        self.raio_chegada = self.get_parameter('raio_chegada').value
        if not 0 < self.distancia_chegada < self.raio_chegada < self.distancia_saida:
            raise ValueError('Use 0 < distancia_chegada < raio_chegada < distancia_saida')
        self.menor_distancia = math.inf
        self.acao_node = SegueLinha()
        self.acao_node.habilitado = False
        self.robot_state = 'segue_linha'
        self.state_machine = {'segue_linha': self.segue_linha, 'done': self.done}
        self.estados_clientes = ['segue_linha']
        self.iniciando = True
        self.origem = None
        self.inicio = time.monotonic()
        self.create_timer(0.05, self.control)

    def segue_linha(self):
        self.acao_node.habilitado = (
            self.x is not None and time.monotonic() - self.ultima_odom < 0.5
        )
        if not self.acao_node.habilitado:
            return
        if self.origem is None:
            self.origem = (self.x, self.y)
            self.inicio = time.monotonic()
            self.acao_node.reset()
            self.get_logger().info('Seguindo a linha.')
        distancia = math.hypot(self.x - self.origem[0], self.y - self.origem[1])
        if distancia > self.distancia_saida:
            self.iniciando = False
        if self.iniciando:
            return
        # A odometria escorrega ao longo da volta, então o robô pode passar
        # ao lado da origem sem chegar a menos de distancia_chegada. Dentro
        # do raio de chegada, para quando a distância começar a aumentar
        # (passou do ponto mais próximo da origem).
        chegou = distancia < self.distancia_chegada
        if distancia < self.raio_chegada:
            self.menor_distancia = min(self.menor_distancia, distancia)
            if distancia > self.menor_distancia + 0.05:
                chegou = True
        if chegou:
            self.acao_node.stop()
            self.robot_state = 'done'
            duracao = time.monotonic() - self.inicio
            self.get_logger().info(
                f'Volta concluída em {duracao:.1f} s (a {distancia:.2f} m da origem).'
            )

    def done(self):
        self.acao_node.stop()

    def control(self):
        self.state_machine[self.robot_state]()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    executor = SingleThreadedExecutor()
    try:
        node = SeguidorDeLinha()
        executor.add_node(node)
        executor.add_node(node.acao_node)
        while rclpy.ok() and node.robot_state != 'done':
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.acao_node.stop()
        executor.shutdown()
        if node is not None:
            node.acao_node.destroy_node()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()