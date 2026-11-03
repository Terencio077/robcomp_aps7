"""Ação PegaCreeper.

Estados:
    buscar -> centralizar -> aproximar -> preparar -> alinhar -> avancar
           -> fechar -> levantar -> done

Por que o robô batia no creeper antes:
  * A aproximação final dependia de a distância da tag chegar a 0.20 m. Perto
    da garra a tag sai da imagem (ou a leitura não chega nesse valor) e a
    garra encosta no creeper antes, empurrando ele. distancia_captura PRECISA
    ser calibrada no robô (ver modo calibrar).
  * O centro da garra não é necessariamente o centro da imagem. Centralizar
    a tag no meio da imagem pode deixar a garra torta e um dedo bate no
    creeper. O parâmetro centro_garra corrige isso.

Agora: com o robô parado a distancia_preparo, mede a distância, alinha, e faz
o avanço final pela odometria (mantendo o rumo), andando exatamente
distancia_medida - distancia_captura. Se a tag ainda estiver visível, ela
também corrige o rumo e pode mandar parar antes.

Calibração (robô parado, garra abre e ombro vai para frente sozinhos):
    ros2 run entregavel_7 pega_creeper --ros-args -p calibrar:=true
Coloque o creeper DENTRO da garra aberta, onde ela deve fechar, e leia d e e
na tela/terminal:  distancia_captura := d   e   centro_garra := e.
"""

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


def limitar(valor, limite):
    return max(-limite, min(limite, valor))


def diferenca_angulo(alvo, atual):
    return math.atan2(math.sin(alvo - atual), math.cos(alvo - atual))


class PegaCreeper(Acao, Garra, Odom):
    def __init__(self):
        Acao.__init__(self, 'pega_creeper_node')
        Garra.__init__(self, delay=0.0)  # esperas sem time.sleep (não trava callbacks)
        Odom.__init__(self)
        self.declare_parameters('', [
            ('kp', 0.5),
            ('angular_max', 0.25),
            ('velocidade', 0.10),
            ('velocidade_final', 0.04),
            ('distancia_preparo', 0.45),
            ('distancia_captura', 0.20),
            ('centro_garra', 0.0),
            ('tolerancia', 0.04),
            ('avanco_maximo', 0.40),
            ('tempo_garra', 3.0),
            ('posicao_ombro', 'frente'),
            ('angulo_giro', 180.0),
            ('velocidade_giro', 1.2),
            ('distancia_recuo', 0.30),
            ('calibrar', False),
        ])
        p = lambda nome: self.get_parameter(nome).value
        self.calibrar = p('calibrar')
        self.kp = p('kp')
        self.angular_max = p('angular_max')
        self.velocidade = p('velocidade')
        self.velocidade_final = p('velocidade_final')
        self.distancia_preparo = p('distancia_preparo')
        self.distancia_captura = p('distancia_captura')
        self.centro_garra = p('centro_garra')
        self.tolerancia = p('tolerancia')
        self.avanco_maximo = p('avanco_maximo')
        self.tempo_garra = p('tempo_garra')
        self.posicao_ombro = p('posicao_ombro')
        self.angulo_giro = math.radians(p('angulo_giro'))
        self.velocidade_giro = p('velocidade_giro')
        self.distancia_recuo = p('distancia_recuo')
        if self.posicao_ombro not in ('frente', 'baixo'):
            raise ValueError('posicao_ombro deve ser frente ou baixo')
        if not 0 < self.distancia_captura < self.distancia_preparo:
            raise ValueError('Use 0 < distancia_captura < distancia_preparo')
        if self.velocidade_final <= 0 or self.tempo_garra <= 0:
            raise ValueError('velocidade_final e tempo_garra devem ser positivos')

        self.bridge = CvBridge()
        self.largura = None
        self.erro = None
        self.distancia = math.inf
        self.cor = None
        self.id = None
        self.tags = []
        self.ultima_imagem = 0.0
        self.ultimas_tags = 0.0
        self.ultimo_alvo = 0.0
        self.alinhado_frames = 0
        self.frames_com_tag = 0
        self.medidas = []
        self.ultimo_diagnostico = 0.0
        self.ultimo_comando_garra = 0.0
        self.comando_atual = None
        self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(
            TaginfoArray, '/tag_list', self.tag_callback, qos_profile_sensor_data
        )
        self.state_machine.update({
            'buscar': self.buscar,
            'centralizar': self.centralizar,
            'aproximar': self.aproximar,
            'preparar': self.preparar,
            'alinhar': self.alinhar,
            'avancar': self.avancar,
            'fechar': self.fechar,
            'levantar': self.levantar,
            'girar': self.girar,
            'baixar': self.baixar,
            'soltar': self.soltar,
            'recuar': self.recuar,
        })
        self.get_logger().info(
            f'PegaCreeper v7 (gira, solta, recua): preparo={self.distancia_preparo:.2f} m, '
            f'captura={self.distancia_captura:.2f} m, centro_garra={self.centro_garra:+.3f}'
            + ('  [MODO CALIBRAR]' if self.calibrar else '')
        )

    # ------------------------------------------------------------ infra
    def reset(self):
        self.cor = None
        self.id = None
        self.erro = None
        self.distancia = math.inf
        self.inicio_acao = time.monotonic()
        self.alinhado_frames = 0
        Acao.reset(self, 'buscar')
        self.mudar_estado('buscar')

    def mudar_estado(self, estado):
        self.robot_state = estado
        self.inicio_estado = time.monotonic()
        self.fase = 0
        self.alinhado_frames = 0
        self.medidas = []
        self.get_logger().info(f'-> {estado}')

    def tempo_no_estado(self):
        return time.monotonic() - self.inicio_estado

    def comando(self, cmd):
        """Publica o comando da garra e repete a cada 0.5 s (caso se perca)."""
        agora = time.monotonic()
        if cmd != self.comando_atual or agora - self.ultimo_comando_garra > 0.5:
            self.controla_garra(cmd)
            self.comando_atual = cmd
            self.ultimo_comando_garra = agora

    def garra_conectada(self):
        return (self.ombro_pub.get_subscription_count() > 0
                and self.garra_pub.get_subscription_count() > 0)

    def control(self):
        agora = time.monotonic()
        if agora - self.ultimo_diagnostico >= 1.0:
            self.ultimo_diagnostico = agora
            self.diagnostico()
        if self.calibrar:
            # Base parada; garra aberta e ombro para frente para posicionar o creeper.
            self.cmd_vel_pub.publish(Twist())
            if self.garra_conectada() and agora - self.ultimo_comando_garra > 0.5:
                self.controla_garra('abrir')
                self.controla_garra(self.posicao_ombro)
                self.ultimo_comando_garra = agora
            return
        if agora - self.inicio_acao > 120.0:
            self.get_logger().error('Tempo limite: ação interrompida.')
            self.stop()
            return
        Acao.control(self)

    def diagnostico(self):
        if self.calibrar:
            visiveis = self.tags_frescas() if self.largura else []
            if not visiveis:
                self.get_logger().info('CALIBRAR: nenhuma tag visível.')
            for t in visiveis:
                self.get_logger().info(
                    f'CALIBRAR: tag {t.id}: d={t.distance:.3f} m  e={self.erro_de(t):+.3f}'
                    '  -> use distancia_captura:=d  centro_garra:=e'
                )
            return
        erro = '--' if self.erro is None else f'{self.erro:+.3f}'
        dist = '--' if not math.isfinite(self.distancia) else f'{self.distancia:.3f} m'
        self.get_logger().info(
            f'estado={self.robot_state} | alvo={self.cor}/{self.id} | erro={erro} | d={dist}'
        )

    # ------------------------------------------------------------ percepção
    def tags_frescas(self):
        if time.monotonic() - self.ultimas_tags >= 0.5:
            return []
        return [t for t in self.tags if math.isfinite(t.distance) and t.distance > 0]

    def erro_de(self, tag):
        metade = self.largura / 2
        return (metade - tag.center.x) / metade

    def tag_callback(self, msg):
        self.tags = list(msg.tags)
        self.ultimas_tags = time.monotonic()

    def image_callback(self, msg):
        imagem = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        agora = time.monotonic()
        self.largura = imagem.shape[1]
        tags = self.tags_frescas()
        coordenadas = [(t.id, (int(t.center.x), int(t.center.y))) for t in tags]
        pares = detect_tagged_creepers(imagem, coordenadas)
        por_id = {t.id: t for t in tags}
        if self.id is None and not self.calibrar:
            self.frames_com_tag = self.frames_com_tag + 1 if tags else 0
            if pares:
                # Preferência: creeper confirmado por cor + tag.
                alvo = min(pares, key=lambda c: abs(c['tag_center'][0] - self.largura / 2))
                self.id, self.cor = alvo['id'], alvo['color']
            elif self.frames_com_tag >= 3:
                # Perto, o corpo colorido sai da imagem (ou fica atrás da garra):
                # trava só pela tag, que é o que guia a aproximação.
                alvo = min(tags, key=lambda t: abs(t.center.x - self.largura / 2))
                self.id, self.cor = alvo.id, '?'
            if self.id is not None:
                self.get_logger().info(f'Alvo travado: creeper {self.cor} id {self.id}')
        tag = por_id.get(self.id)
        if tag is not None:
            # Erro em relação ao centro da GARRA, não ao centro da imagem.
            self.erro = self.erro_de(tag) - self.centro_garra
            self.distancia = float(tag.distance)
            self.ultimo_alvo = agora
        else:
            self.erro = None
            self.distancia = math.inf
        self.ultima_imagem = agora
        self.mostrar_imagem(imagem, coordenadas, pares, por_id)

    def alvo_visivel(self):
        return self.erro is not None and time.monotonic() - self.ultimo_alvo < 0.4

    def mostrar_imagem(self, imagem, coordenadas, pares, por_id):
        altura, largura = imagem.shape[:2]
        cores = {'red': (0, 0, 255), 'green': (0, 180, 0), 'blue': (255, 0, 0)}
        x_garra = int(largura / 2 * (1 - self.centro_garra))
        cv2.line(imagem, (largura // 2, 0), (largura // 2, altura - 1), (255, 255, 255), 1)
        cv2.line(imagem, (x_garra, 0), (x_garra, altura - 1), (0, 255, 255), 1)
        for tag_id, centro in coordenadas:
            cv2.circle(imagem, centro, 8, (0, 255, 255), 2)
            t = por_id[tag_id]
            cv2.putText(imagem, f'ID {tag_id}: d={t.distance:.3f} e={self.erro_de(t):+.3f}',
                        (centro[0] + 5, centro[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
        for par in pares:
            cor = cores[par['color']]
            cv2.circle(imagem, par['center'], 7, cor, -1)
            cv2.line(imagem, par['tag_center'], par['center'], cor, 2)
        erro = '--' if self.erro is None else f'{self.erro:+.3f}'
        dist = '--' if not math.isfinite(self.distancia) else f'{self.distancia:.3f}m'
        estado = 'CALIBRAR (creeper dentro da garra)' if self.calibrar else self.robot_state
        texto = f'{estado} | {self.cor}/{self.id} | e={erro} | d={dist}'
        cv2.rectangle(imagem, (0, 0), (largura - 1, 28), (0, 0, 0), -1)
        cv2.putText(imagem, texto, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow('Pega Creeper', imagem)
        cv2.waitKey(1)

    # ------------------------------------------------------------ estados
    def buscar(self):
        # Espera achar um creeper (cor + tag, ou só a tag).
        if self.id is not None and self.alvo_visivel():
            self.mudar_estado('centralizar')
            return
        # Fica parado esperando: o creeper é colocado na frente do robô.

    def centralizar(self):
        if not self.alvo_visivel():
            if time.monotonic() - self.ultimo_alvo > 1.5:
                self.mudar_estado('buscar')
            return
        if abs(self.erro) < self.tolerancia:
            # Já está alinhado: não mexe (zona morta).
            self.alinhado_frames += 1
        else:
            self.alinhado_frames = 0
            self.twist.angular.z = limitar(self.kp * self.erro, self.angular_max)
        if self.alinhado_frames >= 5:
            self.mudar_estado('aproximar')

    def aproximar(self):
        if not self.alvo_visivel():
            if time.monotonic() - self.ultimo_alvo > 1.5:
                self.mudar_estado('centralizar')
            return
        if abs(self.erro) > 0.15:
            self.mudar_estado('centralizar')
            return
        if self.distancia <= self.distancia_preparo:
            self.mudar_estado('preparar')
            return
        if abs(self.erro) >= self.tolerancia:
            self.twist.angular.z = limitar(self.kp * self.erro, self.angular_max)
        self.twist.linear.x = min(
            self.velocidade, max(0.03, 0.5 * (self.distancia - self.distancia_preparo))
        )

    def preparar(self):
        # Robô parado (twist já é zero). Abre a garra, depois ombro para frente.
        if not self.garra_conectada():
            if self.tempo_no_estado() > 3.0 and self.fase == 0:
                self.get_logger().warn('Ninguém escutando os tópicos da garra!')
                self.fase = -1
            return
        if self.fase <= 0:
            self.comando('abrir')
            self.fase = 1
            self.inicio_estado = time.monotonic()
        elif self.fase == 1:
            self.comando('abrir')
            if self.tempo_no_estado() >= self.tempo_garra:
                self.fase = 2
                self.inicio_estado = time.monotonic()
        elif self.fase == 2:
            self.comando(self.posicao_ombro)
            if self.tempo_no_estado() >= self.tempo_garra:
                self.mudar_estado('alinhar')

    def alinhar(self):
        # Parado, com a garra pronta: mede a distância até o creeper.
        if not self.alvo_visivel():
            if self.tempo_no_estado() > 3.0:
                self.get_logger().error('Perdi o creeper ao preparar a garra. Parando.')
                self.stop()
            return
        if time.monotonic() - self.ultima_odom > 0.5 or self.yaw is None:
            return
        # Só mede a distância (média de vários quadros). Não gira: o
        # alinhamento já foi feito uma vez no centralizar.
        self.alinhado_frames += 1
        self.medidas.append(self.distancia)
        if self.alinhado_frames < 6:
            return
        self.twist = Twist()
        medidas = sorted(self.medidas)
        distancia = medidas[len(medidas) // 2]
        self.avanco = distancia - self.distancia_captura
        if self.avanco > self.avanco_maximo:
            self.get_logger().warn('Ainda longe; voltando a aproximar.')
            self.mudar_estado('aproximar')
            return
        self.avanco = max(0.0, self.avanco)
        self.origem_final = (self.x, self.y)
        self.yaw_final = self.yaw
        self.get_logger().info(
            f'd medida={distancia:.3f} m -> avanço final de {self.avanco:.3f} m por odometria'
        )
        self.mudar_estado('avancar')

    def avancar(self):
        if time.monotonic() - self.ultima_odom > 0.5:
            return  # sem odometria: fica parado
        if self.tempo_no_estado() > self.avanco / self.velocidade_final + 10.0:
            self.get_logger().error('Avanço demorou demais. Fechando a garra mesmo assim.')
            self.mudar_estado('fechar')
            return
        percorrido = math.hypot(self.x - self.origem_final[0], self.y - self.origem_final[1])
        restante = self.avanco - percorrido
        if self.alvo_visivel():
            # Se a tag ainda aparece, ela também pode mandar parar antes.
            restante = min(restante, self.distancia - self.distancia_captura)
        # Avanço final 100% reto: o alinhamento já foi feito parado no estado
        # alinhar. Perto da garra a leitura da tag fica ruim e fazia o robô
        # girar do nada, então aqui não há nenhuma correção angular.
        self.twist.angular.z = 0.0
        if restante <= 0.0:
            self.twist = Twist()
            self.mudar_estado('fechar')
            return
        self.twist.linear.x = min(self.velocidade_final, max(0.015, 0.8 * restante))

    def fechar(self):
        self.comando('fechar')
        if self.tempo_no_estado() >= self.tempo_garra:
            self.mudar_estado('levantar')

    def levantar(self):
        self.comando('cima')
        if self.tempo_no_estado() >= self.tempo_garra:
            self.get_logger().info('Creeper capturado e levantado.')
            self.mudar_estado('girar')

    def girar(self):
        # Gira rápido no lugar, segurando o creeper, usando o yaw da odometria.
        if time.monotonic() - self.ultima_odom > 0.5 or self.yaw is None:
            return
        if self.fase == 0:
            self.yaw_alvo = self.yaw + self.angulo_giro
            self.fase = 1
        falta = diferenca_angulo(self.yaw_alvo, self.yaw)
        if abs(falta) < 0.05 or self.tempo_no_estado() > 15.0:
            self.twist = Twist()
            self.mudar_estado('baixar')
            return
        velocidade = max(0.3, min(self.velocidade_giro, 2.0 * abs(falta)))
        self.twist.angular.z = math.copysign(velocidade, falta)

    def baixar(self):
        # Desce o ombro até a posição em que pegou (creeper encosta no chão).
        self.comando(self.posicao_ombro)
        if self.tempo_no_estado() >= self.tempo_garra:
            self.mudar_estado('soltar')

    def soltar(self):
        self.comando('abrir')
        if self.tempo_no_estado() >= self.tempo_garra:
            self.mudar_estado('recuar')

    def recuar(self):
        # Anda de ré em linha reta e para longe do creeper.
        if time.monotonic() - self.ultima_odom > 0.5:
            return
        if self.fase == 0:
            self.origem_recuo = (self.x, self.y)
            self.fase = 1
        andado = math.hypot(self.x - self.origem_recuo[0], self.y - self.origem_recuo[1])
        if andado >= self.distancia_recuo or self.tempo_no_estado() > 15.0:
            self.twist = Twist()
            self.get_logger().info('Creeper entregue no chão. Fim da ação.')
            self.stop()
            return
        self.twist.linear.x = -0.10


def main(args=None):
    try:
        executar(PegaCreeper, args)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()  