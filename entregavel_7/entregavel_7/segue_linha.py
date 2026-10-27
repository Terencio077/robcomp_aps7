import time

import cv2
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

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
            ('roi_inicio', 0.80),
            ('roi_fim', 0.98),
            ('salto_maximo', 0.20),
        ])
        self.time_attack = self.get_parameter('time_attack').value
        self.velocidade = self.get_parameter('velocidade').value
        self.velocidade_ataque = self.get_parameter('velocidade_ataque').value
        self.kp = self.get_parameter('kp').value
        self.angular_max = self.get_parameter('angular_max').value
        self.lado = self.get_parameter('lado_externo').value
        if self.lado not in ('direita', 'esquerda'):
            raise ValueError('lado_externo deve ser direita ou esquerda')
        self.roi_inicio = self.get_parameter('roi_inicio').value
        self.roi_fim = self.get_parameter('roi_fim').value
        self.salto_maximo = self.get_parameter('salto_maximo').value
        if not 0 <= self.roi_inicio < self.roi_fim <= 1:
            raise ValueError('Use 0 <= roi_inicio < roi_fim <= 1')
        if not 0 < self.salto_maximo <= 1:
            raise ValueError('salto_maximo deve estar em (0, 1]')
        self.cx_anterior = None
        self.cy_linha = None
        self.cx_linha = None
        self.w = None
        self.erro = None
        self.ultima_imagem = 0.0
        self.habilitado = True
        self.bridge = CvBridge()
        self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback,
            qos_profile_sensor_data
        )
        self.ultimo_log = 0.0
        self.get_logger().info('SegueLinha: CAMERA COMPRIMIDA, ROI PROXIMA e continuidade da linha.')
        self.state_machine['seguir'] = self.seguir

    def reset(self):
        self.cx_anterior = None
        Acao.reset(self, 'seguir')

    def image_callback(self, msg):
        imagem = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        altura, largura = imagem.shape[:2]
        y1, y2 = int(altura * self.roi_inicio), int(altura * self.roi_fim)
        roi = imagem[y1:y2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, (18, 90, 80), (38, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidatos = []
        faixa = max(3, int(altura * 0.025))
        for contorno in contornos:
            if cv2.contourArea(contorno) < mascara.size * 0.005:
                continue
            x, y, w, h = cv2.boundingRect(contorno)
            # Usa apenas a ponta inferior do trecho, não seu centro inteiro.
            # Assim um segmento comprido à frente não puxa o alvo da curva.
            componente = mascara.copy()
            componente.fill(0)
            cv2.drawContours(componente, [contorno], -1, 255, cv2.FILLED)
            inicio = max(y, y + h - faixa)
            m = cv2.moments(componente[inicio:y + h], binaryImage=True)
            if m['m00']:
                candidatos.append({
                    'cx': m['m10'] / m['m00'],
                    'cy': y1 + inicio + m['m01'] / m['m00'],
                    'base': y + h,
                })
        self.cx_linha = None
        self.cy_linha = None
        self.w = largura / 2
        self.erro = None
        if candidatos:
            # Dentre os trechos que chegam mais perto da base da imagem,
            # acompanha aquele que continua perto do alvo da imagem anterior.
            base = max(c['base'] for c in candidatos)
            proximos = [c for c in candidatos if c['base'] >= base - faixa]
            alvo = None
            if self.cx_anterior is None:
                escolher = max if self.lado == 'direita' else min
                alvo = escolher(proximos, key=lambda c: c['cx'])
            else:
                continuos = [c for c in proximos
                             if abs(c['cx'] - self.cx_anterior) <= largura * self.salto_maximo]
                if continuos:
                    alvo = min(continuos, key=lambda c: abs(c['cx'] - self.cx_anterior))
            if alvo is not None:
                self.cx_linha = alvo['cx']
                self.cy_linha = alvo['cy']
                self.cx_anterior = self.cx_linha
                self.erro = (self.w - self.cx_linha) / self.w
        # Se só sobrou outro trecho longe do anterior, para em vez de trocar de pista.
        self.ultima_imagem = time.monotonic()
        cv2.rectangle(imagem, (0, y1), (largura - 1, y2), (255, 0, 0), 2)
        cv2.line(imagem, (largura // 2, y1), (largura // 2, y2), (255, 255, 255), 2)
        if self.cx_linha is not None:
            cv2.circle(imagem, (int(self.cx_linha), int(self.cy_linha)), 7, (0, 0, 255), -1)
        texto = 'SEM LINHA' if self.erro is None else f'erro={self.erro:.3f} lado={self.lado}'
        cv2.rectangle(imagem, (0, 0), (largura - 1, 30), (0, 0, 0), -1)
        cv2.putText(imagem, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        cv2.imshow('Segue Linha', imagem)
        cv2.imshow('Mascara Amarela', mascara)
        cv2.waitKey(1)

    def control(self):
        agora = time.monotonic()
        if agora - self.ultimo_log >= 1.0:
            self.ultimo_log = agora
            erro = '--' if self.erro is None else f'{self.erro:.3f}'
            self.get_logger().info(
                f'imagem={agora - self.ultima_imagem < 0.5} | '
                f'linha={self.erro is not None} | erro={erro} | '
                f'odom_habilita={self.habilitado}'
            )
        Acao.control(self)

    def destroy_node(self):
        cv2.destroyAllWindows()
        return super().destroy_node()

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
