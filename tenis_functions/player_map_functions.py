import cv2
import numpy as np

def h_inv(H):
  return np.linalg.inv(H)

def image_to_court(point_img, H_inv):
    """
    point_img: np.array shape (1,1,2) o (1,2)
    """
    pt = point_img.reshape(1, 1, 2)
    pt_court = cv2.perspectiveTransform(pt, H_inv)
    return pt_court.reshape(2)

def get_court_border(court_ref):
    """
    Devuelve los 4 puntos del contorno de la pista
    en coordenadas de pista plana
    """
    h, w = court_ref.shape[:2]

    border = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    return border

def project_court_border(border_pts, H):
    """
    border_pts: (4,2) en pista
    H: homografía pista → imagen
    """
    pts = border_pts.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H)
    return projected.reshape(-1, 2).astype(int)

def draw_court_border(image, border_img_pts, color=(0, 255, 255), thickness=3):
    """
    Dibuja el contorno de la pista en la imagen
    """
    img = image.copy()

    cv2.polylines(
        img,
        [border_img_pts],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    return img

# ======================================================
# 1) Construir minimapa EXACTO: pista + margen extendido (sin padding extra)
# ======================================================
def build_minicourt_extended_base(court_ref, scale=0.3, extended_margin_px=80):
    """
    Devuelve:
      base: imagen BGR negra del tamaño EXACTO del área extendida (sin pad)
      ext: margen extendido (px en coords court_ref)
      scale: escala usada
      court_ref_img: court_ref en uint8 (H,W)
    """
    if isinstance(court_ref, tuple):
        court_ref = court_ref[0]
    court_ref_img = court_ref

    # Asegurar tipo correcto
    if court_ref_img.dtype != np.uint8:
        court_ref_img = court_ref_img.astype(np.uint8)

    h, w = court_ref_img.shape[:2]
    ext = int(extended_margin_px)

    # Tamaño del área extendida (coords court_ref)
    ext_w = w + 2 * ext
    ext_h = h + 2 * ext

    # Tamaño en píxeles del minimapa
    mini_w = int(round(ext_w * scale))
    mini_h = int(round(ext_h * scale))

    # Base negra (se usará como “fondo” a mezclar con la imagen real)
    base = np.zeros((mini_h, mini_w, 3), dtype=np.uint8)

    return base, ext, scale, court_ref_img


# ======================================================
# 2) Dibujar líneas de pista dentro del minimapa extendido (sin márgenes extra)
# ======================================================
def draw_court_lines_on_minimap(minimap, court_ref_img, ext, scale, color=(255,255,255), thickness=1):
    """
    Dibuja la pista (court_ref) dentro del minimapa extendido.
    Se coloca desplazada por ext (porque el origen extendido es -ext,-ext).
    """
    # Convertir court_ref a BGR y escalarlo
    court_bgr = cv2.cvtColor(court_ref_img, cv2.COLOR_GRAY2BGR)
    court_scaled = cv2.resize(
        court_bgr,
        (int(round(court_ref_img.shape[1] * scale)), int(round(court_ref_img.shape[0] * scale))),
        interpolation=cv2.INTER_AREA
    )

    ox = int(round(ext * scale))
    oy = int(round(ext * scale))

    # Pegarlo en el minimapa extendido
    h_s, w_s = court_scaled.shape[:2]
    minimap[oy:oy+h_s, ox:ox+w_s] = court_scaled

    # Reforzar líneas a blanco puro (opcional, para que se vean más)
    gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    minimap[mask] = color

    # Marco del área extendida (sin pad)
    cv2.rectangle(minimap, (0, 0), (minimap.shape[1]-1, minimap.shape[0]-1), color, 2)

    return minimap


# ======================================================
# 3) Mapear punto court -> minimapa extendido
# ======================================================
def court_point_to_overlay_xy(pt, court_ref_shape, ext, scale, margin_xy):
    """
    Convierte un punto en coords court_ref a coords (x,y) en la imagen final,
    teniendo en cuenta:
      - ext (margen extendido en coords court_ref)
      - scale (escala del minimapa)
      - margin_xy (x0,y0) donde se pegó el minimapa en la imagen final
    """
    x, y = float(pt[0]), float(pt[1])
    x0, y0 = margin_xy

    mx = int(round((x + ext) * scale)) + x0
    my = int(round((y + ext) * scale)) + y0
    return mx, my



# ======================================================
# 4) Lógica: fuera del área extendida y clamp simple
# ======================================================
def is_outside_extended_area(pt, court_shape, margin_px=80):
    x, y = float(pt[0]), float(pt[1])
    h, w = court_shape[:2]
    M = float(margin_px)
    return not (-M <= x < w + M and -M <= y < h + M)

def clamp_to_extended_rect(pt, court_shape, margin_px=80):
    x, y = float(pt[0]), float(pt[1])
    h, w = court_shape[:2]
    M = float(margin_px)
    x = max(-M, min(w - 1 + M, x))
    y = max(-M, min(h - 1 + M, y))
    return x, y


def draw_players_on_overlay(
    overlay,
    court_ref_shape,
    player_positions_court,
    ext,
    scale,
    margin_xy=(40, 40),
    extended_margin_px=200,
    r=6
):
    """
    Verde: dentro del área extendida
    Rojo: fuera del área extendida (se clamp-ea al borde extendido)
    Se dibuja en la imagen final (overlay), por encima del minimapa ya mezclado.
    """
    h, w = court_ref_shape[:2]
    M = float(extended_margin_px)

    def outside_ext(pt):
        x, y = float(pt[0]), float(pt[1])
        return not (-M <= x < w + M and -M <= y < h + M)

    def clamp_ext(pt):
        x, y = float(pt[0]), float(pt[1])
        x = max(-M, min(w - 1 + M, x))
        y = max(-M, min(h - 1 + M, y))
        return x, y

    for pt in player_positions_court:
        if pt is None or not hasattr(pt, "__iter__") or len(pt) != 2:
            continue

        x, y = float(pt[0]), float(pt[1])

        if outside_ext((x, y)):
            x, y = clamp_ext((x, y))
            px, py = court_point_to_overlay_xy((x, y), court_ref_shape, ext, scale, margin_xy)
            cv2.circle(overlay, (px, py), r, (0, 0, 255), -1)  # rojo
        else:
            px, py = court_point_to_overlay_xy((x, y), court_ref_shape, ext, scale, margin_xy)
            cv2.circle(overlay, (px, py), r, (0, 255, 0), -1)  # verde

    return overlay



# ======================================================
# 6) Pegar minimapa con fondo semitransparente sobre la imagen real
# ======================================================
def paste_minimap_transparent(image, minimap, x0=40, y0=40, alpha_bg=0.55):
    """
    alpha_bg controla cuánto “tapa” el minimapa:
      0.0 = no se ve
      1.0 = tapa totalmente
    """
    overlay = image.copy()
    h, w = minimap.shape[:2]

    # Recortar si se sale del frame
    H_img, W_img = overlay.shape[:2]
    x1 = min(x0 + w, W_img)
    y1 = min(y0 + h, H_img)

    roi = overlay[y0:y1, x0:x1]
    mini_crop = minimap[0:(y1-y0), 0:(x1-x0)]

    # Mezcla SOLO del fondo/estructura del minimapa con la imagen real
    blended = cv2.addWeighted(roi, 1 - alpha_bg, mini_crop, alpha_bg, 0)
    overlay[y0:y1, x0:x1] = blended

    return overlay


# ======================================================
# 7) Función principal: minimapa sin margen extra + fondo transparente
# ======================================================
def draw_minicourt_overlay(
    image,
    court_ref,
    player_positions_court,
    scale=0.3,
    margin=40,
    extended_margin_px=200,
    alpha_bg=0.55
):
    # 1) Construir minimapa (base extendida)
    minimap, ext, sc, court_ref_img = build_minicourt_extended_base(
        court_ref=court_ref,
        scale=scale,
        extended_margin_px=extended_margin_px
    )

    # 2) Dibujar solo la pista en el minimapa (fondo + líneas)
    minimap = draw_court_lines_on_minimap(
        minimap=minimap,
        court_ref_img=court_ref_img,
        ext=ext,
        scale=sc,
        color=(255, 255, 255),
        thickness=1
    )

    # 3) Mezclar minimapa con transparencia sobre la imagen real
    overlay = paste_minimap_transparent(
        image=image,
        minimap=minimap,
        x0=margin,
        y0=margin,
        alpha_bg=alpha_bg
    )

    # 4) Dibujar jugadores NÍTIDOS encima (sin blending)
    overlay = draw_players_on_overlay(
        overlay=overlay,
        court_ref_shape=court_ref_img.shape,
        player_positions_court=player_positions_court,
        ext=ext,
        scale=sc,
        margin_xy=(margin, margin),
        extended_margin_px=extended_margin_px,
        r=6
    )

    # 5) Título
    cv2.putText(
        overlay,
        "Court View",
        (margin, margin - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return overlay


