import cv2

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from .base_action import Acao
from .linha import localizar_linha, erros


class SegueLinha(Acao):
    def __init__(self):
        super().__init__('segue_linha_node')

        self.declare_parameters('', [
            ('velocidade', 0.10),
            ('kp', 0.5),
            ('angular_max', 0.5),
            ('lado', 'direita'),
            ('visualizar', False),
        ])

        parametro = lambda nome: self.get_parameter(nome).value
        self.velocidade = parametro('velocidade')
        self.kp = parametro('kp')
        self.angular_max = parametro('angular_max')
        self.lado = parametro('lado')
        self.visualizar = parametro('visualizar')

        self.bridge = CvBridge()
        self.erro = None
        self.x_anterior = None
        self.largura_anterior = None
        self.habilitado = False

        self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.state_machine['seguir'] = self.seguir

    def reset(self):
        self.erro = None
        self.x_anterior = None
        self.largura_anterior = None
        Acao.reset(self, 'seguir')

    def seguir(self):
        if not self.habilitado or self.erro is None:
            return

        self.twist.linear.x = self.velocidade
        rotacao = self.kp * self.erro
        self.twist.angular.z = max(
            -self.angular_max,
            min(self.angular_max, rotacao),
        )

    def image_callback(self, msg):
        imagem = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')

        resultado, mascara, y1 = localizar_linha(
            imagem,
            x_anterior=self.x_anterior,
            lado=self.lado,
            largura_anterior=self.largura_anterior,
        )

        if resultado is None:
            self.erro = None
        else:
            self.erro, _, _ = erros(resultado)
            self.x_anterior = resultado['x_base']
            self.largura_anterior = resultado['largura_base']

        if self.visualizar:
            exibicao = imagem.copy()
            meio = exibicao.shape[1] // 2
            cv2.line(
                exibicao,
                (meio, 0),
                (meio, exibicao.shape[0] - 1),
                (255, 0, 0),
                2,
            )

            if resultado is not None:
                pontos = resultado['pontos']
                for x, y in pontos:
                    cv2.circle(exibicao, (int(x), int(y)), 5, (0, 255, 0), -1)

                texto = f'erro: {self.erro:.2f}'
            else:
                texto = 'linha nao detectada'

            cv2.putText(
                exibicao,
                texto,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow('Segue Linha', exibicao)
            cv2.waitKey(1)