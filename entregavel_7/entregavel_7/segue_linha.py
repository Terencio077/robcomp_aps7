import time

import cv2
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .base_action import Acao, executar


class SegueLinha(Acao):
    def __init__(self):
        Acao.__init__(self, 'seguelinha_node')
        self.declare_parameters('', [
            ('time_attack', False),
            ('velocidade', 0.12),
            ('velocidade_ataque', 0.22),
            ('kp', 0.8),
            ('angular_max', 0.5),
            ('lado_externo', 'direita'),
        ])
        self.time_attack = self.get_parameter('time_attack').value
        self.velocidade = self.get_parameter('velocidade').value
        self.velocidade_ataque = self.get_parameter('velocidade_ataque').value
        self.kp = self.get_parameter('kp').value
        self.angular_max = self.get_parameter('angular_max').value
        self.lado = self.get_parameter('lado_externo').value
        if self.lado not in ('direita', 'esquerda'):
            raise ValueError('lado_externo deve ser direita ou esquerda')
        self.erro = None
        self.ultima_imagem = 0.0
        self.habilitado = True
        self.bridge = CvBridge()
        self.create_subscription(
            Image, '/camera/image_raw', self.image_callback,
            qos_profile_sensor_data
        )
        self.state_machine['seguir'] = self.seguir

    def reset(self):
        Acao.reset(self, 'seguir')

    def image_callback(self, msg):
        imagem = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        altura, largura = imagem.shape[:2]
        roi = imagem[int(altura * 0.65):int(altura * 0.75)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, (18, 90, 80), (38, 255, 255))
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        centros = []
        for contorno in contornos:
            if cv2.contourArea(contorno) < mascara.size * 0.005:
                continue
            m = cv2.moments(contorno)
            if m['m00']:
                centros.append(m['m10'] / m['m00'])
        self.erro = None
        if centros:
            cx = max(centros) if self.lado == 'direita' else min(centros)
            self.erro = (largura / 2 - cx) / (largura / 2)
        self.ultima_imagem = time.monotonic()

    def seguir(self):
        if (not self.habilitado or self.erro is None
                or time.monotonic() - self.ultima_imagem > 0.5):
            return
        self.twist.angular.z = max(
            -self.angular_max, min(self.angular_max, self.kp * self.erro)
        )
        self.twist.linear.x = self.velocidade
        if self.time_attack:
            self.twist.linear.x = self.velocidade_ataque * max(
                0.35, 1.0 - abs(self.erro)
            )


def main(args=None):
    executar(SegueLinha, args)


if __name__ == '__main__':
    main()
