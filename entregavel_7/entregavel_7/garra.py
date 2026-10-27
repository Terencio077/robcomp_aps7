from std_msgs.msg import Float64
import time

class Garra:
    def __init__(self, delay: float = 2.0):
        self.delay = delay

        self.ombro_pub = self.create_publisher(
            Float64,
            "/joint1_position_controller/command",
            10
        )
        self.garra_pub = self.create_publisher(
            Float64,
            "/joint2_position_controller/command",
            10
        )

    def controla_garra(self, command: str):
        if command == 'abrir':
            publisher, valor = self.garra_pub, -1.0
        elif command == 'fechar':
            publisher, valor = self.garra_pub, 0.0
        elif command == 'cima':
            publisher, valor = self.ombro_pub, 1.5
        elif command == 'frente':
            publisher, valor = self.ombro_pub, 0.0
        elif command == 'baixo':
            publisher, valor = self.ombro_pub, -1.0
        else:
            raise ValueError(f"Comando desconhecido para a garra: {command!r}")

        msg = Float64()
        msg.data = valor
        publisher.publish(msg)
        time.sleep(self.delay)
