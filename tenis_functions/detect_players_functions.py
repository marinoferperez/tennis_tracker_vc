import torch
import cv2
import numpy as np
import itertools
import glob
import matplotlib.pyplot as plt
import random
from player_map_functions import h_inv, image_to_court

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

# función que define los kp virtuales manualmente a partir de los reales

def kp_virtuales(court_keypoints):
    # parte superior de la pista

    # se almacenan los kps reales más relevantes, que
    # corresponden a las zonas donde más presente está el jugador superior en
    # un clip
    top_left  = court_keypoints[0]
    top_right = court_keypoints[1]

    sub_top_left = court_keypoints[4]
    sub_top_right = court_keypoints[6]

    mid_top = court_keypoints[12]

    # se calcula el punto medio de la linea final superior para pista de dobles
    kp_top = midpoint(top_left, top_right)
    # se calcula el punto medio de la linea final superior para pista individual
    kp_sub_top = midpoint(sub_top_left, sub_top_right)

    # se calcula el punto intermedio en los medios de la linea final superior
    kp_parte_superior = midpoint(kp_top, kp_sub_top)

    # se calcula el punto intermedio final entre la parte superior y el punto
    # intermedio de la mitad de la pista superior
    kp_parte_superior = midpoint(kp_parte_superior, mid_top)

    # parte inferior de la pista

    # se almacenan los kps reales más relevantes, que
    # corresponden a las zonas donde más presente está el jugador inferior en
    # un clip
    bot_left  = court_keypoints[2]
    bot_right = court_keypoints[3]

    sub_bot_left = court_keypoints[5]
    sub_bot_right = court_keypoints[7]

    mid_bottom = court_keypoints[13]

    # se calcula el punto medio de la linea final inferior para pista de dobles
    kp_bot = midpoint(bot_left, bot_right)
    # se calcula el punto medio de la linea final inferior para pista individual
    kp_sub_bot = midpoint(sub_bot_left, sub_bot_right)

    # se calcula el punto intermedio en los medios de la linea final inferior
    kp_parte_inferior = midpoint(kp_bot, kp_sub_bot)

    # se calcula el punto intermedio final entre la parte inferior y el punto
    # intermedio de la mitad de la pista inferior
    kp_parte_inferior = midpoint(kp_parte_inferior, mid_bottom)

    return kp_parte_superior, kp_parte_inferior

# función que dibuja las bounding box a partir de los pares de coordenadas
# devueltos por yolo tras detectar una persona u objeto

def dibuja_bbox(img, persona, label, color):
    x1, y1, x2, y2 = map(int, persona["bbox"])

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        f"{label} ({persona['conf']:.2f})",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return img

# funcion que dibuja las bounding box para los dos tenistas detectados

def dibuja_bbox_jugadores(img, top_player, bottom_player):
    img_out = img.copy()

    # Jugador superior
    img_out = dibuja_bbox(
        img_out,
        top_player,
        label="Jugador 2",
        color=(0, 255, 0)
    )

    # Jugador inferior
    img_out = dibuja_bbox(
        img_out,
        bottom_player,
        label="Jugador 1",
        color=(255, 0, 0)
    )

    return img_out


# funcion que calcula el punto de la bounding box que simula el centro de los pies
# del jugador
def bbox_center_pies(bb):
  return (int((bb[2] + bb[0]) / 2) , int(bb[3]))


# función que asigna los jugadores superior e inferior utilizando la posicion
# de los pies de cada jugador.

def asigna_jugadores_por_pies(players, kp_top_court, kp_bottom_court, H):
    # a partir de la homografía, se calcula la inversa (imagen -> pista)
    H_inv = h_inv(H)

    # si no hay al menos dos jugadores, no es posible asignar ambos
    if len(players) < 2:
        return players[0], None

    best_cost = float("inf")
    best_pair = None

    # se prueban todos los pares de jugadores distintos
    for i in range(len(players)):
        for j in range(len(players)):
            if i == j:
                continue

            # --- pies de jugadores --> pista ---

            # se proyectan los pies del jugador candidato superior
            p_top = image_to_court(
                np.array([players[i]["x_c"], players[i]["y_c"]], dtype=np.float32),
                H_inv
            )

            # se proyectan los pies del jugador candidato inferior
            p_bot = image_to_court(
                np.array([players[j]["x_c"], players[j]["y_c"]], dtype=np.float32),
                H_inv
            )

            # calcula el coste en coord de pista
            cost = (
                np.linalg.norm(p_top - kp_top_court) +
                np.linalg.norm(p_bot - kp_bottom_court)
            )

            # se actualiza el coste 
            if cost < best_cost:
                best_cost = cost
                best_pair = (players[i], players[j])

    return best_pair

# rellena las posiciones del jugador en los frames donde no fue detectado utilizando
# las detecciones validas de frames cercanos

def interpola_posiciones(datos_trackeo, jugador):
    # jugador = "top_player" o "bottom_player"

    frames = [] # lista con los índices de frame
    posiciones = [] # lista de posiciones (cx, cy) o (nan, nan) si no se detecta

    for d in datos_trackeo:
        frames.append(d["frame_idx"])

        if d[jugador] is not None:
            posiciones.append(d[jugador])
        else:
            posiciones.append((np.nan, np.nan))

    # convertimos a arrays
    frames = np.array(frames)
    posiciones = np.array(posiciones)

    # necesario ya que np.interp utiliza vectores 1d, no 2d (cx, cy)
    x = posiciones[:, 0] # separamos las coordenadas quedandonos con todos los cxs
    y = posiciones[:, 1] # separamos las coordenadas quedandonos con todos los cys

    # interpolamos linealmente ignorando los nones
    valid = ~np.isnan(x)

    # si por algún motivo no hay detecciones válidas, abortamos
    if valid.sum() < 2:
        raise ValueError(f"No hay suficientes detecciones válidas para interpolar '{jugador}'")

    x_interp = np.interp(frames, frames[valid], x[valid])
    y_interp = np.interp(frames, frames[valid], y[valid])

    # se vuelven a unir los pares de coordenadas ya interpolados
    posiciones_interpoladas = []
    for i in range(len(x_interp)):
        posiciones_interpoladas.append((x_interp[i], y_interp[i]))

    return posiciones_interpoladas

#### METRICAS: DISTANCIA ACUMULADA

ALTURA_JUGADOR_ESTIMADA = 1.80

# función que convierte una distancia medida en píxeles a metros
# usando una altura de referencia conocida (en metros y píxeles)
def distancia_pixeles_a_metros(distancia_px, ref_altura_m, ref_altura_px):
    return (distancia_px * ref_altura_m) / ref_altura_px

# función que conviernte una distancia en metros a píxeles usando la misma
# relación de proporcionalidad basada en la altura del jugador
def metros_a_distancia_pixeles(metros, ref_altura_m, ref_altura_px):
    return (metros * ref_altura_px) / ref_altura_m

# función que calcula la altura de una bbox en px
def altura_bbox_px(bbox):
    x1, y1, x2, y2 = bbox
    return max(1, (y2 - y1))  # max(1, ...) evita división por 0

# función que calcula una referencia de altura en píxeles para cada frame
# a partir de las bounding boxes detectadas
def calcula_ref_altura_px(datos_trackeo, key_bbox):
    N = len(datos_trackeo)
    ref = np.full(N, np.nan, dtype=np.float32)

    # si se detecta bbox, nos quedamos con su altura en px
    for i in range(N):
        bbox = datos_trackeo[i].get(key_bbox, None)
        if bbox is not None:
            ref[i] = altura_bbox_px(bbox)

    # si no se detecta se le atribuye el ultimo valor de altura valido
    ultimo_ref = np.nan
    for i in range(N):
        if not np.isnan(ref[i]):
            ultimo_ref = ref[i]
        else:
            ref[i] = ultimo_ref

    # si los primeros frames no tienen detección,
    # se rellena con la primera altura válida encontrada
    primera_valida = None
    for i in range(N):
        if not np.isnan(ref[i]):
            primera_valida = ref[i]
            break

    if primera_valida is None:
        raise ValueError(f"No hay ninguna bbox válida en '{key_bbox}'")

    for i in range(N):
        if np.isnan(ref[i]):
            ref[i] = primera_valida
        else:
            break

    return ref

# función que calcula la distancia acumulada frame a frame
def calcula_distancia_acumulada(posis_interp, refs_altura_px):
    distancia_acumulada = np.zeros(len(posis_interp), dtype=np.float32)

    total = 0.0
    for i in range(1, len(posis_interp)):
        # posicion del jugador en el frame anterior
        x1, y1 = posis_interp[i-1]
        # posicion del jugador en el frame actual
        x2, y2 = posis_interp[i]

        # dist recorrida entre ambos frames en px
        dist_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # conversion de px a mtros utilizando la altura estimada 
        # y la altura de ref en px
        dist_m = distancia_pixeles_a_metros(
            dist_px,
            ALTURA_JUGADOR_ESTIMADA,
            refs_altura_px[i]
        )

        # se acumula la distanncia recorrida
        total += dist_m
        distancia_acumulada[i] = total

    return distancia_acumulada

# funcion auxiliar que dibuja el hud en el que se muestran las distancias de ambos jugadores
def dibuja_hud(top_dist, bottom_dist, img, alpha=0.6):
    _, W = img.shape[:2]
    
    hud_x = W - 420
    hud_y = 20
    hud_w = 400
    hud_h = 90

    # fondo semitransparente
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + hud_h),
        (0, 0, 0),
        -1
    )

    img_out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # texto
    cv2.putText(
        img_out,
        f"Top player: {top_dist:.2f} m",
        (hud_x + 20, hud_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img_out,
        f"Bottom player: {bottom_dist:.2f} m",
        (hud_x + 20, hud_y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    return img_out

# funcion que recoge el pipeline para el trackeo de la dist acumulada
def trackeo_distancias_acumuladas(modelo, detector, img_paths, court_points, recalc_num_frames=7):
    # lista donde vamos a guardar la info del top/bottom player en cada frame
    datos_trackeo = []

    H = None
    court_keypoints = None

    for frame_idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # almacena:
        # numero de frame, (cx_top, cy_top), (cx_bottom, cy_bottom), bbox_top, bbox_bottom
        # si en un frame no detecta al jugador -> none (es lo que nos interesa saber)
        frame_info = {
            "frame_idx": frame_idx,
            "top_player": None,        # (x_pies, y_pies)
            "bottom_player": None,     # (x_pies, y_pies)
            "top_bbox": None,
            "bottom_bbox": None
        }

        # se obtienen y refinan los kp reales + homografia cada 7 frames
        if frame_idx % recalc_num_frames == 0 or H is None:
            try:
                H, court_keypoints = process_court_image(
                    img_path,
                    detector,
                    crop_size=110
                )
            except Exception:
                H = None
                court_keypoints = None

        if H is None or court_keypoints is None:
            datos_trackeo.append(frame_info)
            continue

        # deteccion de personas con yolo
        predicciones = modelo(img_rgb, conf=0.4, classes=[0], verbose=False)

        bboxes_players = []
        for r in predicciones:
            for bbox in r.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy()
                bboxes_players.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

        if len(bboxes_players) < 2:
            datos_trackeo.append(frame_info)
            continue

        # calculo de pies para cada bbox registrada
        for bb in bboxes_players:
            bb["x_c"], bb["y_c"] = bbox_center_pies(bb["bbox"])

        # calculo de los kp virtuales
        kp_top_court, kp_bottom_court = kp_virtuales(court_points)

        # seleccion de jugadores mediante kp virtuales y h inversa
        top_player, bottom_player = asigna_jugadores_por_pies(
            bboxes_players,
            kp_top_court,
            kp_bottom_court,
            H
        )

        if top_player is not None:
            # guardamos el centro del jug superior para interpolar los frames
            frame_info["top_player"] = (top_player["x_c"], top_player["y_c"])
            # guardamos las bbox 
            frame_info["top_bbox"] = top_player["bbox"]

        if bottom_player is not None:
            # guardamos el centro del jug inferior para interpolar los frames
            frame_info["bottom_player"] = (bottom_player["x_c"], bottom_player["y_c"])
            # guardamos las bbox 
            frame_info["bottom_bbox"] = bottom_player["bbox"]

        datos_trackeo.append(frame_info)

    # calculo de las alturas de referencia en pixeles para cada jugador segun bbox
    top_ref_altura_px = calcula_ref_altura_px(datos_trackeo, "top_bbox")
    bottom_ref_altura_px = calcula_ref_altura_px(datos_trackeo, "bottom_bbox")
    
    # interpolacion de posiciones
    top_interp = interpola_posiciones(datos_trackeo, "top_player")
    bottom_interp = interpola_posiciones(datos_trackeo, "bottom_player")

    # calculo de la dist acumulada en cada frame
    top_dist_acumulada = calcula_distancia_acumulada(
        top_interp,
        top_ref_altura_px
    )

    bottom_dist_acumulada = calcula_distancia_acumulada(
        bottom_interp,
        bottom_ref_altura_px
    )

    return top_dist_acumulada, bottom_dist_acumulada, datos_trackeo