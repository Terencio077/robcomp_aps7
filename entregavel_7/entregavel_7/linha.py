"""Detecção da linha amarela usada pela ação SegueLinha.

1. Considera só os trechos amarelos que chegam perto da base da imagem.
2. Entre eles, escolhe o que continua a linha do quadro anterior.
3. Percorre esse trecho em fatias horizontais, da base para cima, sempre
   pegando o segmento mais próximo do ponto da fatia de baixo.
4. Devolve os pontos da linha: o mais baixo dá a posição e o mais alto
   antecipa a curva.
"""

import cv2
import numpy as np


def _trechos(colunas, gap=3):
    """Agrupa colunas ocupadas em trechos contínuos [(inicio, fim), ...]."""
    if colunas.size == 0:
        return []
    quebras = np.nonzero(np.diff(colunas) > gap)[0]
    inicios = np.concatenate(([colunas[0]], colunas[quebras + 1]))
    fins = np.concatenate((colunas[quebras], [colunas[-1]]))
    return list(zip(inicios.tolist(), fins.tolist()))


def mascara_amarela(bgr, hsv_min=(18, 90, 80), hsv_max=(38, 255, 255)):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, np.array(hsv_min, np.uint8), np.array(hsv_max, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara


def localizar_linha(bgr, x_anterior=None, lado='direita', roi_inicio=0.5,
                    largura_anterior=None,
                    fatias=8, salto=0.12, faixa_base=0.45, preferir_lado=True,
                    hsv_min=(18, 90, 80), hsv_max=(38, 255, 255)):
    """Encontra a linha que o robô está seguindo.

    Retorna (resultado, mascara, y1). resultado é None se não achou linha,
    senão um dict com 'pontos' [(x, y), ...] da base para cima (coordenadas
    da imagem inteira), 'x_base' e 'largura'.
    """
    altura, largura = bgr.shape[:2]
    y1 = int(altura * roi_inicio)
    mascara = mascara_amarela(bgr[y1:], hsv_min, hsv_max)
    rh = mascara.shape[0]

    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    area_min = max(40, mascara.size * 0.002)
    faixa = max(4, rh // 12)
    limite_base = rh * (1.0 - faixa_base)

    candidatos = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < area_min:
            continue
        base = y + h
        # Pedaço de pista que não chega perto do robô: é "o próximo ponto
        # amarelo" lá na frente. Ignora.
        if base < limite_base:
            continue
        _, xs = np.nonzero(rotulos[max(y, base - faixa):base] == i)
        if xs.size == 0:
            continue
        candidatos.append((i, base, float(xs.mean())))

    if not candidatos:
        return None, mascara, y1

    if x_anterior is None:
        mais_baixa = max(c[1] for c in candidatos)
        perto = [c for c in candidatos if c[1] >= mais_baixa - 2 * faixa]
        escolher = max if lado == 'direita' else min
        escolhido = escolher(perto, key=lambda c: c[2])
    else:
        # Continuidade com o quadro anterior; desempata pelo que está mais
        # perto do robô (base mais baixa).
        escolhido = min(
            candidatos,
            key=lambda c: abs(c[2] - x_anterior) + 0.5 * largura * (rh - c[1]) / rh
        )

    indice, base, x_base = escolhido
    componente = rotulos == indice
    topo = stats[indice][1]
    altura_fatia = rh / fatias
    pontos = []
    # A primeira fatia usa a posição e a largura da linha no quadro anterior:
    # se a junção de dois ramos já chegou na base da imagem, o centro do
    # trecho fundido não desloca o ponto.
    x_ref = x_base if x_anterior is None else x_anterior
    largura_ref = largura_anterior
    largura_base = None

    # Trechos (runs) de cada fatia, da base para cima.
    fatias_trechos = []
    for k in range(fatias):
        yb = int(round(rh - k * altura_fatia))
        ya = int(round(rh - (k + 1) * altura_fatia))
        if yb <= topo:
            break
        if ya >= base:
            fatias_trechos.append((ya, yb, None))
            continue
        colunas = np.nonzero(componente[ya:yb].any(axis=0))[0]
        fatias_trechos.append((ya, yb, _trechos(colunas)))

    def perto_de(trechos, x):
        return [t for t in trechos
                if t[0] - salto * largura <= x <= t[1] + salto * largura]

    for k, (ya, yb, trechos) in enumerate(fatias_trechos):
        if trechos is None:
            continue
        validos = perto_de(trechos, x_ref)
        if not validos:
            break
        if len(validos) > 1 and preferir_lado:
            # Bifurcação com os dois ramos já separados: fica no lado externo.
            escolher = max if lado == 'direita' else min
            ini, fim = escolher(validos, key=lambda t: (t[0] + t[1]) / 2)
        else:
            ini, fim = min(validos, key=lambda t: abs((t[0] + t[1]) / 2 - x_ref))
        largura_trecho = fim - ini + 1
        cx = (ini + fim) / 2
        if largura_ref is None:
            largura_ref = largura_trecho
        meia = largura_ref / 2
        # Por causa da perspectiva, a fita afina de fatia em fatia. Um trecho
        # bem mais largo que o da fatia de baixo não é só a linha.
        if largura_trecho > 1.2 * largura_ref + 4:
            # Curva fechada, junção de dois ramos ou cruzamento. O centro do
            # trecho é a média dos ramos, então decide para onde ir:
            folga = 0.2 * largura_ref + 3
            esquerda = (x_ref - meia) - ini > folga
            direita = fim - (x_ref + meia) > folga
            acima = None
            for _, _, t in fatias_trechos[k + 1:]:
                if t:
                    acima = t
                    break
            continua_reto = bool(acima) and any(
                t[0] - largura_ref <= x_ref <= t[1] + largura_ref for t in acima
            )
            lado_externo_aberto = direita if lado == 'direita' else esquerda
            if preferir_lado and lado_externo_aberto and not (esquerda and direita):
                # Abre para o lado externo: segue por ele.
                cx = fim - meia if lado == 'direita' else ini + meia
            elif continua_reto:
                # Há linha seguindo reto logo acima (ramo interno ou
                # cruzamento): não vira.
                cx = min(max(x_ref, ini + meia), fim - meia)
            elif esquerda != direita and acima is not None:
                # Nada reto acima: é curva. Acompanha a linha até a ponta.
                cx = ini + meia if esquerda else fim - meia
            else:
                cx = min(max(x_ref, ini + meia), fim - meia)
        else:
            if largura_trecho > largura_ref:
                # Um pouco mais largo que o esperado: pega a janela da largura
                # da linha mais próxima do ponto anterior (não deixa um ramo
                # que está se juntando puxar o centro aos poucos).
                cx = min(max(x_ref, ini + meia), fim - meia)
                largura_ref = min(largura_trecho, 1.05 * largura_ref + 2)
            else:
                largura_ref = largura_trecho
        if largura_base is None:
            largura_base = largura_ref
        pontos.append((float(cx), float(y1 + (ya + yb) / 2)))
        x_ref = cx

    if not pontos:
        return None, mascara, y1

    return {'pontos': pontos, 'x_base': pontos[0][0], 'largura': largura,
            'largura_base': largura_base}, mascara, y1


def erros(resultado, peso_longe=0.6):
    """Erro normalizado em [-1, 1]. Positivo = linha à esquerda (gira p/ esquerda)."""
    meio = resultado['largura'] / 2
    pontos = resultado['pontos']
    erro_perto = (meio - pontos[0][0]) / meio
    erro_longe = (meio - pontos[-1][0]) / meio
    if len(pontos) < 3:
        peso_longe = 0.0
    return (1 - peso_longe) * erro_perto + peso_longe * erro_longe, erro_perto, erro_longe
