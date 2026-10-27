import math
import time

import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

from .base_action import Acao, executar
from .garra import Garra
from .odom import Odom
from .vision import segment_creepers


class PegaCreeper(Acao, Garra, Odom):
    def __init__(self):
        Acao.__init__(self, 'pega_creeper_node')
        Garra.__init__(self, delay=0.0)
        Odom.__init__(self)
        self.declare_parameters('', [
            ('kp', 0.6),
            ('velocidade', 0.08),
            ('velocidade_final', 0.03),
            ('distancia_preparo', 0.35),
            ('distancia_captura', 0.20),
            ('tempo_garra', 2.0),
        ])
        self.kp = self.get_parameter('kp').value
        self.velocidade = self.get_parameter('velocidade').value
        self.velocidade_final = self.get_parameter('velocidade_final').value
        self.distancia_preparo = self.get_parameter('distancia_preparo').value
        self.distancia_captura = self.get_parameter('distancia_captura').value
        self.tempo_garra = self.get_parameter('tempo_garra').value
        if not (0 < self.distancia_captura < self.distancia_preparo
                and self.distancia_preparo - self.distancia_captura <= 0.25):
            raise ValueError('Use captura < preparo, com diferença de até 0.25 m')
        if self.velocidade_final <= 0 or self.tempo_garra <= 0:
            raise ValueError('velocidade_final e tempo_garra devem ser positivos')
        self.bridge = CvBridge()
        self.erro = None
        self.cor = None
        self.cx = None
        self.distancia = math.inf
        self.ultima_imagem = 0.0
        self.ultimo_scan = 0.0
        self.create_subscription(
            Image, '/camera/image_raw', self.image_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, '/scan', self.laser_callback, qos_profile_sensor_data
        )
        self.state_machine.update({
            'centralizar': self.centralizar,
            'aproximar': self.aproximar,
            'preparar': self.preparar,
            'avancar': self.avancar,
            'fechar': self.fechar,
            'levantar': self.levantar,
        })

    def reset(self):
        self.cor = None
        self.cx = None
        self.erro = None
        self.comando_enviado = False
        Acao.reset(self, 'centralizar')

    def mudar_estado(self, estado):
        self.robot_state = estado
        self.comando_enviado = False
        self.inicio_estado = time.monotonic()
        self.get_logger().info(estado)

    def image_callback(self, msg):
        imagem = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        candidatos = segment_creepers(imagem)
        if self.cor is not None:
            candidatos = [c for c in candidatos if c['color'] == self.cor]
        self.erro = None
        if candidatos:
            if self.cx is None:
                alvo = max(candidatos, key=lambda c: c['bbox'][2] * c['bbox'][3])
            else:
                alvo = min(candidatos, key=lambda c: abs(c['center'][0] - self.cx))
            self.cor = alvo['color']
            self.cx = alvo['center'][0]
            metade = imagem.shape[1] / 2
            self.erro = (metade - self.cx) / metade
        self.ultima_imagem = time.monotonic()

    def laser_callback(self, msg):
        distancias = np.asarray(msg.ranges)
        angulos = msg.angle_min + np.arange(len(distancias)) * msg.angle_increment
        angulos = (angulos + np.pi) % (2 * np.pi) - np.pi
        validos = (
            (np.abs(angulos) <= np.deg2rad(5))
            & np.isfinite(distancias)
            & (distancias >= msg.range_min)
            & (distancias <= msg.range_max)
            & (distancias > 0)
        )
        self.distancia = float(np.min(distancias[validos])) if validos.any() else math.inf
        self.ultimo_scan = time.monotonic()

    def alvo_visivel(self):
        return self.erro is not None and time.monotonic() - self.ultima_imagem < 0.5

    def centralizar(self):
        if not self.alvo_visivel():
            return
        self.twist.angular.z = max(-0.3, min(0.3, self.kp * self.erro))
        if abs(self.erro) < 0.05:
            self.twist = Twist()
            self.mudar_estado('aproximar')

    def aproximar(self):
        if not self.alvo_visivel():
            return
        agora = time.monotonic()
        if (agora - self.ultimo_scan > 0.5 or agora - self.ultima_odom > 0.5
                or not math.isfinite(self.distancia)):
            return
        if abs(self.erro) > 0.12:
            self.mudar_estado('centralizar')
            return
        if self.distancia <= self.distancia_preparo:
            self.avanco = max(0.0, self.distancia - self.distancia_captura)
            self.origem_final = (self.x, self.y)
            self.mudar_estado('preparar')
            return
        self.twist.angular.z = max(-0.3, min(0.3, self.kp * self.erro))
        self.twist.linear.x = min(
            self.velocidade, max(0.03, 0.5 * (self.distancia - self.distancia_preparo))
        )

    def preparar(self):
        if not self.comando_enviado:
            if (self.ombro_pub.get_subscription_count() == 0
                    or self.garra_pub.get_subscription_count() == 0):
                return
            self.cmd_vel_pub.publish(Twist())
            self.controla_garra('baixo')
            self.controla_garra('abrir')
            self.comando_enviado = True
            self.inicio_estado = time.monotonic()
        if time.monotonic() - self.inicio_estado >= self.tempo_garra:
            self.mudar_estado('avancar')

    def avancar(self):
        agora = time.monotonic()
        if agora - self.inicio_estado > self.avanco / self.velocidade_final + 8.0:
            self.get_logger().error('Avanço interrompido: captura não concluída.')
            self.stop()
            return
        if agora - self.ultima_odom > 0.5 or agora - self.ultima_imagem > 0.5:
            return
        percorrido = math.hypot(
            self.x - self.origem_final[0], self.y - self.origem_final[1]
        )
        if percorrido >= self.avanco:
            self.mudar_estado('fechar')
            return
        if self.erro is not None:
            self.twist.angular.z = max(-0.2, min(0.2, self.kp * self.erro))
            if abs(self.erro) > 0.12:
                return
        self.twist.linear.x = self.velocidade_final

    def fechar(self):
        if not self.comando_enviado:
            self.cmd_vel_pub.publish(Twist())
            self.controla_garra('fechar')
            self.comando_enviado = True
            self.inicio_estado = time.monotonic()
        if time.monotonic() - self.inicio_estado >= self.tempo_garra:
            self.mudar_estado('levantar')

    def levantar(self):
        if not self.comando_enviado:
            self.cmd_vel_pub.publish(Twist())
            self.controla_garra('cima')
            self.comando_enviado = True
            self.inicio_estado = time.monotonic()
        if time.monotonic() - self.inicio_estado >= self.tempo_garra:
            self.get_logger().info('Sequência de captura finalizada.')
            self.stop()


def main(args=None):
    executar(PegaCreeper, args)


if __name__ == '__main__':
    main()
