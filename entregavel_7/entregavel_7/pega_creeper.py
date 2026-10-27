import math
import time

import cv2

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from robcomp_interfaces.msg import TaginfoArray

from .base_action import Acao, executar
from .garra import Garra
from .odom import Odom
from .vision import detect_tagged_creepers


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
            ('calibrar', False),
        ])
        self.calibrar = self.get_parameter('calibrar').value
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
        self.alinhado_frames = 0
        self.captura_frames = 0
        self.cor = None
        self.id = None
        self.tags = []
        self.cx = None
        self.distancia = math.inf
        self.ultima_imagem = 0.0
        self.ultimas_tags = 0.0
        self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            TaginfoArray, '/tag_list', self.tag_callback, qos_profile_sensor_data
        )
        self.ultimo_diagnostico = 0.0
        self.get_logger().info('PegaCreeper: detector da APS6, camera comprimida e ombro FRENTE (0.0).')
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
        self.id = None
        self.cx = None
        self.erro = None
        self.distancia = math.inf
        self.inicio_acao = time.monotonic()
        self.comando_enviado = False
        self.alinhado_frames = 0
        self.captura_frames = 0
        Acao.reset(self, 'centralizar')

    def control(self):
        agora = time.monotonic()
        if agora - self.ultimo_diagnostico >= 1.0:
            self.ultimo_diagnostico = agora
            erro = '--' if self.erro is None else f'{self.erro:.3f}'
            distancia = '--' if not math.isfinite(self.distancia) else f'{self.distancia:.3f} m'
            self.get_logger().info(
                f'estado={self.robot_state} | alvo={self.cor}/{self.id} | '
                f'erro={erro} | distancia={distancia} | '
                f'imagem={agora - self.ultima_imagem < 0.5} | '
                f'tags={agora - self.ultimas_tags < 0.5} | '
                f'odom={agora - self.ultima_odom < 0.5}'
            )
        if self.calibrar:
            # Observação sem movimento da base e sem comandos automáticos dos servos.
            self.twist = Twist()
            self.cmd_vel_pub.publish(self.twist)
            return
        if agora - self.inicio_acao > 120.0:
            self.get_logger().error('Tempo limite: ação interrompida sem confirmar captura.')
            self.stop()
            return
        Acao.control(self)

    def mudar_estado(self, estado):
        self.robot_state = estado
        self.comando_enviado = False
        self.inicio_estado = time.monotonic()
        self.get_logger().info(estado)


    def tag_callback(self, msg):
        self.tags = list(msg.tags)
        self.ultimas_tags = time.monotonic()

    def image_callback(self, msg):
        imagem = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        agora = time.monotonic()
        tags = [t for t in self.tags
                if math.isfinite(t.distance) and t.distance > 0]
        if agora - self.ultimas_tags >= 0.5:
            tags = []
        coordenadas = [(t.id, (int(t.center.x), int(t.center.y))) for t in tags]
        # Mesma função e mesmos limiares usados na APS 6 enviada pelo grupo.
        pares = detect_tagged_creepers(imagem, coordenadas)
        por_id = {t.id: t for t in tags}
        self.erro = None
        self.distancia = math.inf
        if self.id is None and pares:
            # Escolhe um creeper confirmado por cor + marcador à frente.
            alvo = min(pares, key=lambda c: abs(c['tag_center'][0] - imagem.shape[1] / 2))
            self.id = alvo['id']
            self.cor = alvo['color']
        # Após identificar a cor, acompanha o mesmo ID. A região colorida pode
        # sair do enquadramento perto da garra sem invalidar a tag ainda visível.
        tag = por_id.get(self.id)
        if tag is not None:
            self.cx = tag.center.x
            self.distancia = float(tag.distance)
            metade = imagem.shape[1] / 2
            self.erro = (metade - self.cx) / metade
        self.ultima_imagem = agora
        alinhado = self.erro is not None and abs(self.erro) < 0.05
        self.alinhado_frames = self.alinhado_frames + 1 if alinhado else 0
        capturavel = (alinhado and math.isfinite(self.distancia)
                      and self.distancia <= self.distancia_captura)
        self.captura_frames = self.captura_frames + 1 if capturavel else 0
        self.mostrar_imagem(imagem, coordenadas, pares)

    def mostrar_imagem(self, imagem, tags, pares):
        altura, largura = imagem.shape[:2]
        cores = {'red': (0, 0, 255), 'green': (0, 180, 0), 'blue': (255, 0, 0)}
        cv2.line(imagem, (largura // 2, 0), (largura // 2, altura - 1), (255, 255, 255), 1)
        por_id = {t.id: t for t in self.tags}
        for tag_id, centro in tags:
            cv2.circle(imagem, centro, 8, (0, 255, 255), 2)
            tag = por_id[tag_id]
            cv2.putText(imagem, f'ID {tag_id}: {tag.distance:.3f}m',
                        (centro[0] + 5, centro[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
        for par in pares:
            cor = cores[par['color']]
            cv2.circle(imagem, par['center'], 7, cor, -1)
            cv2.line(imagem, par['tag_center'], par['center'], cor, 2)
            pos = (min(par['tag_center'][0], par['center'][0]),
                   max(45, min(par['tag_center'][1], par['center'][1]) - 10))
            cv2.putText(imagem, f"ID {par['id']} - {par['color']}", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        erro = '--' if self.erro is None else f'{self.erro:.3f}'
        distancia = '--' if not math.isfinite(self.distancia) else f'{self.distancia:.3f}m'
        estado = 'CALIBRAR: BASE PARADA' if self.calibrar else self.robot_state
        texto = f'{estado} | {self.cor}/{self.id} | e={erro} | d={distancia}'
        cv2.rectangle(imagem, (0, 0), (largura - 1, 28), (0, 0, 0), -1)
        cv2.putText(imagem, texto, (5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        cv2.imshow('Pega Creeper', imagem)
        cv2.waitKey(1)

    def alvo_visivel(self):
        return (self.erro is not None
                and time.monotonic() - self.ultima_imagem < 0.5
                and time.monotonic() - self.ultimas_tags < 0.5)

    def centralizar(self):
        if not self.alvo_visivel():
            return
        self.twist.angular.z = max(-0.3, min(0.3, self.kp * self.erro))
        if self.alinhado_frames >= 5:
            self.twist = Twist()
            self.mudar_estado('aproximar')

    def aproximar(self):
        if not self.alvo_visivel():
            return
        agora = time.monotonic()
        if (agora - self.ultimas_tags > 0.5 or agora - self.ultima_odom > 0.5
                or not math.isfinite(self.distancia)):
            return
        if abs(self.erro) > 0.12:
            self.mudar_estado('centralizar')
            return
        if self.distancia <= self.distancia_preparo:
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
            self.controla_garra('abrir')
            self.fase_preparo = 'abrindo'
            self.comando_enviado = True
            self.inicio_estado = time.monotonic()
            return
        if time.monotonic() - self.inicio_estado < self.tempo_garra:
            return
        if self.fase_preparo == 'abrindo':
            # Frente = 0.0. Não manda o ombro ao chão (-1.0).
            self.controla_garra('frente')
            self.fase_preparo = 'posicionando'
            self.inicio_estado = time.monotonic()
            return
        if not self.alvo_visivel() or time.monotonic() - self.ultima_odom >= 0.5:
            return
        # Mede novamente depois de mover o ombro, com o robô parado.
        self.distancia_inicio_final = self.distancia
        self.avanco = max(0.0, self.distancia - self.distancia_captura)
        if self.avanco > 0.25:
            self.get_logger().error('Objeto longe demais para o avanço final. Reinicie a aproximação.')
            self.stop()
            return
        self.origem_final = (self.x, self.y)
        self.captura_frames = 0
        self.mudar_estado('avancar')
        self.get_logger().info(
            f'Distância após preparar={self.distancia:.3f} m; '
            f'captura={self.distancia_captura:.3f} m; avanço esperado={self.avanco:.3f} m'
        )

    def avancar(self):
        agora = time.monotonic()
        if agora - self.inicio_estado > self.avanco / self.velocidade_final + 15.0:
            self.get_logger().error('Avanço interrompido: confira o alvo e calibre a distância da garra.')
            self.stop()
            return
        # Sem confirmação visual, para; não fecha por cumprir um deslocamento.
        if not self.alvo_visivel() or agora - self.ultima_odom > 0.5:
            return
        self.twist.angular.z = max(-0.12, min(0.12, self.kp * self.erro))
        if abs(self.erro) > 0.05:
            return
        if self.distancia <= self.distancia_captura:
            self.twist = Twist()
            if self.captura_frames >= 3:
                self.mudar_estado('fechar')
            return
        percorrido = math.hypot(
            self.x - self.origem_final[0], self.y - self.origem_final[1]
        )
        # Odometria limita o avanço; a distância atual da tag autoriza fechar.
        if percorrido > min(0.25, self.avanco + 0.03):
            self.get_logger().error('Distância da tag e avanço divergiram. Parando sem fechar.')
            self.stop()
            return
        self.twist.linear.x = min(
            self.velocidade_final, max(0.005, 0.4 * (self.distancia - self.distancia_captura))
        )

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
            self.get_logger().info('Sequência finalizada. Confirme visualmente a captura.')
            self.stop()


def main(args=None):
    try:
        executar(PegaCreeper, args)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()