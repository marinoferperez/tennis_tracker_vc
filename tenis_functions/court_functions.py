import torch
import cv2
import numpy as np
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights


class CourtLineDetector:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v3_small(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = torch.nn.Linear(in_features, 28)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_bgr):
        """
        image_bgr: imagen OpenCV (H,W,3) en BGR
        devuelve keypoints en píxeles originales
        """
        h, w = image_bgr.shape[:2]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints_480 = outputs.squeeze().cpu().numpy()

        keypoints = keypoints_480.copy()
        keypoints[0::2] *= w / 480.0
        keypoints[1::2] *= h / 480.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        """
        Dibuja los keypoints sobre la imagen
        """
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                image,
                str(i // 2),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Aplica los keypoints a una lista de frames
        """
        output_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame.copy(), keypoints)
            output_frames.append(frame)
        return output_frames


def crop_square(image, center_x, center_y, size):
    """
    Recorta un cuadrado de la imagen.

    Args:
        image: imagen BGR (H, W, C)
        center_x, center_y: centro del recorte
        size: lado del cuadrado (en píxeles)

    Returns:
        Imagen recortada (size x size x C)
    """
    h, w = image.shape[:2]
    half = size // 2

    x1 = max(center_x - half, 0)
    y1 = max(center_y - half, 0)
    x2 = min(center_x + half, w)
    y2 = min(center_y + half, h)

    crop = image[y1:y2, x1:x2].copy()
    return crop


def zhang_suen_thinning(binary):
    img = binary.copy() // 255
    changed = True

    while changed:
        changed = False
        to_remove = []

        rows, cols = img.shape
        for step in [0, 1]:
            to_remove.clear()

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    P = [
                        img[i, j],
                        img[i-1, j], img[i-1, j+1], img[i, j+1], img[i+1, j+1],
                        img[i+1, j], img[i+1, j-1], img[i, j-1], img[i-1, j-1]
                    ]

                    if P[0] != 1:
                        continue

                    neighbors = sum(P[1:])
                    if neighbors < 2 or neighbors > 6:
                        continue

                    transitions = sum(
                        (P[k] == 0 and P[k+1] == 1)
                        for k in range(1, 8)
                    ) + (P[8] == 0 and P[1] == 1)

                    if transitions != 1:
                        continue

                    if step == 0:
                        if P[1] * P[3] * P[5] != 0:
                            continue
                        if P[3] * P[5] * P[7] != 0:
                            continue
                    else:
                        if P[1] * P[3] * P[7] != 0:
                            continue
                        if P[1] * P[5] * P[7] != 0:
                            continue

                    to_remove.append((i, j))

            for i, j in to_remove:
                img[i, j] = 0
                changed = True

    return (img * 255).astype(np.uint8)


def preprocess_for_hough(image):
    """
    Prepara la imagen para Hough:
    - Reduce colores
    - Elimina fondo dominante
    - Engrosa líneas lejanas
    - Afina con Zhang-Suen
    """
    # --- 1. Reducir colores (KMeans simple) ---
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    _, labels, centers = cv2.kmeans(
        Z, 4, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    reduced = centers[labels.flatten()].reshape(image.shape)

    # --- 2. Eliminar colores dominantes (fondo) ---
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    # --- 3. Engrosar líneas (dilatación adaptativa) ---
    h, w = binary.shape
    kernel_size = max(1, min(h, w) // 150)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    thick = cv2.dilate(binary, kernel, iterations=1)

    # --- 4. Afinar con Zhang–Suen ---
    thinned = zhang_suen_thinning(thick)

    return thinned


def detect_lines(image):
    pre = preprocess_for_hough(image)

    lines = cv2.HoughLinesP(
      pre,
      rho=1,
      theta=np.pi / 180,
      threshold=15,
      minLineLength=10,
      maxLineGap=15
    )
    lines = np.squeeze(lines)
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines

#junto las lineas detectadas por hough_lines para devolver simplemente dos lineas y calcular su punto de intersección
def merge_lines(lines, dist_thresh=25, angle_thresh=5):
    if len(lines) == 0:
        return []

    used = [False] * len(lines)
    merged = []

    for i, l1 in enumerate(lines):
        if used[i]:
            continue

        x1,y1,x2,y2 = l1
        angle1 = np.degrees(np.arctan2(y2-y1, x2-x1))

        pts = [(x1,y1), (x2,y2)]
        used[i] = True

        for j, l2 in enumerate(lines):
            if used[j]:
                continue

            x3,y3,x4,y4 = l2
            angle2 = np.degrees(np.arctan2(y4-y3, x4-x3))

            if abs(angle1 - angle2) > angle_thresh:
                continue

            # distancia mínima entre segmentos (aprox por centros)
            c1 = np.array([(x1+x2)/2, (y1+y2)/2])
            c2 = np.array([(x3+x4)/2, (y3+y4)/2])

            if np.linalg.norm(c1 - c2) > dist_thresh:
                continue

            pts.extend([(x3,y3), (x4,y4)])
            used[j] = True

        # ---- reconstrucción colineal ----
        pts = np.array(pts, dtype=np.float32)

        # vector director
        v = np.array([np.cos(np.radians(angle1)), np.sin(np.radians(angle1))])

        # proyección escalar sobre la dirección
        proj = pts @ v

        p_min = pts[np.argmin(proj)]
        p_max = pts[np.argmax(proj)]

        merged.append([
            int(p_min[0]), int(p_min[1]),
            int(p_max[0]), int(p_max[1])
        ])

    return merged

def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    img = image.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def intersect(l1, l2):
    """ Encuentra intersección (x,y) de dos líneas """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None
    px = int(((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom)
    py = int(((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom)
    return [px, py]

import itertools
def closest_intersection_to_center(lines, crop_shape):
    """
    lines: lista de líneas [x1,y1,x2,y2]
    crop_shape: crop.shape
    """
    if len(lines) < 2:
        return None

    h, w = crop_shape[:2]
    center = np.array([w / 2, h / 2], dtype=np.float32)

    intersections = []

    for l1, l2 in itertools.combinations(lines, 2):
        pt = intersect(l1, l2)
        if pt is None:
            continue

        # solo puntos dentro del crop
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            intersections.append(pt)

    if len(intersections) == 0:
        return None

    # elegir la más cercana al centro
    dists = [np.linalg.norm(pt - center) for pt in intersections]
    best_idx = np.argmin(dists)

    return intersections[best_idx]


def create_tennis_court_reference(scale=50, line_thickness=5): # <--- AUMENTADO A 5
    """
    Crea modelo de pista con líneas más gruesas para soportar la perspectiva lejana.
    """
    # Dimensiones oficiales (metros)
    COURT_WIDTH  = 10.97
    COURT_HEIGHT = 23.77
    SINGLES_MARGIN = 1.37
    SERVICE_DIST = 6.40

    W = int(COURT_WIDTH * scale)
    H = int(COURT_HEIGHT * scale)

    court = np.zeros((H, W), dtype=np.uint8)

    # Función auxiliar
    def draw_line(p1, p2):
        pt1 = (int(p1[0] * scale), int(p1[1] * scale))
        pt2 = (int(p2[0] * scale), int(p2[1] * scale))
        cv2.line(court, pt1, pt2, 255, line_thickness)

    # 1. PERÍMETRO Y SINGLES
    draw_line((0, 0), (0, COURT_HEIGHT))
    draw_line((COURT_WIDTH, 0), (COURT_WIDTH, COURT_HEIGHT))
    draw_line((0, 0), (COURT_WIDTH, 0))
    draw_line((0, COURT_HEIGHT), (COURT_WIDTH, COURT_HEIGHT))

    draw_line((SINGLES_MARGIN, 0), (SINGLES_MARGIN, COURT_HEIGHT))
    draw_line((COURT_WIDTH - SINGLES_MARGIN, 0), (COURT_WIDTH - SINGLES_MARGIN, COURT_HEIGHT))

    # 2. RED Y CUADROS DE SAQUE
    net_y = COURT_HEIGHT / 2
    draw_line((0, net_y), (COURT_WIDTH, net_y)) # Red

    y_top = net_y - SERVICE_DIST
    y_bot = net_y + SERVICE_DIST

    # Líneas horizontales de saque
    draw_line((SINGLES_MARGIN, y_top), (COURT_WIDTH - SINGLES_MARGIN, y_top))
    draw_line((SINGLES_MARGIN, y_bot), (COURT_WIDTH - SINGLES_MARGIN, y_bot))

    # 3. LÍNEA CENTRAL (La que no se veía)
    # Importante: La dibujamos un poco más gruesa extra si es necesario,
    # o simplemente confiamos en el grosor general aumentado.
    center_x = COURT_WIDTH / 2

    # Truco: Dibujarla en dos segmentos (arriba y abajo de la red)
    # a veces ayuda a visualizar si algo tapa el centro, pero es lo mismo matemáticamente.
    draw_line((center_x, y_top), (center_x, y_bot))

    # Puntos de referencia (TL, TR, BR, BL)
    ref_pts = np.array([
        # 1–4: esquinas exteriores
        [0, 0],
        [COURT_WIDTH, 0],
        [0, COURT_HEIGHT],
        [COURT_WIDTH, COURT_HEIGHT],

        # 5–8: líneas de singles
        [SINGLES_MARGIN, 0],
        [SINGLES_MARGIN, COURT_HEIGHT],
        [COURT_WIDTH - SINGLES_MARGIN, 0],
        [COURT_WIDTH - SINGLES_MARGIN, COURT_HEIGHT],

        # 9–12: intersecciones líneas de saque
        [SINGLES_MARGIN, y_top],
        [COURT_WIDTH - SINGLES_MARGIN, y_top],
        [SINGLES_MARGIN, y_bot],
        [COURT_WIDTH - SINGLES_MARGIN, y_bot],

        # 13–14: centros de línea de saque
        [center_x, y_top],
        [center_x, y_bot],
    ], dtype=np.float32) * scale

    return court, ref_pts


#calcula la homografia entre los puntos encontrados y los puntos de la pista de referencia
def compute_homography(image_points, ref_points):

    H, mask = cv2.findHomography(
        ref_points,
        image_points,
        cv2.RANSAC,
        ransacReprojThreshold=5.0
    )
    return H

def overlay_court(image, court_ref, H):
    # Usar INTER_CUBIC o INTER_LANCZOS4 ayuda a mantener nitidez en transformaciones extremas
    warped = cv2.warpPerspective(
        court_ref,
        H,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC
    )

    overlay = image.copy()

    # Umbral bajo para captar líneas que quedaron grises/tenues por la lejanía
    mask = warped > 10

    # Dibujar en amarillo (o el color que quieras)
    overlay[mask] = (0, 255, 255)

    return overlay

def process_court_image(image_path, detector, crop_size = 100 , line_thickness = 5):
    """
    Processes a tennis court image by detecting and refining keypoints,
    computing homography, and overlaying a court reference.
    """
    # 1. Load the image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None
    h, w = image_bgr.shape[:2]

    # 3. Predict initial keypoints
    initial_keypoints_flat = detector.predict(image_bgr)

    # 4. Convert initial_keypoints into (x, y) pairs
    initial_keypoint_pairs = initial_keypoints_flat.reshape(-1, 2)

    # 5. Initialize lists for crops, crop_centers, and refined points
    crops = []
    crop_centers = []
    refined_points_local = [] # Store local refined points (or None)

    # 6. Iterate through each (x, y) keypoint for refinement
    for i, (x, y) in enumerate(initial_keypoint_pairs):
        cx = int(round(x))
        cy = int(round(y))

        crop = crop_square(image_bgr, cx, cy, crop_size)
        crops.append(crop)
        crop_centers.append((cx, cy))

        lines = detect_lines(crop)
        merged_lines = merge_lines(lines)
        pt_local_refined = closest_intersection_to_center(merged_lines, crop.shape)
        refined_points_local.append(pt_local_refined)

    # 7. Translate refined points (local) back to global image coordinates
    refined_keypoints_global_list = []
    half_crop_size = crop_size / 2.0

    for i, pt_crop in enumerate(refined_points_local):
        cx, cy = crop_centers[i]
        if pt_crop is None:
            # If no refinement, use the original predicted keypoint
            x_global = initial_keypoint_pairs[i][0]
            y_global = initial_keypoint_pairs[i][1]
        else:
            x_crop, y_crop = pt_crop
            # Translate local crop coordinates to global image coordinates
            x_global = cx + (x_crop - half_crop_size)
            y_global = cy + (y_crop - half_crop_size)

        refined_keypoints_global_list.append((x_global, y_global))

    refined_keypoints_global = np.array(refined_keypoints_global_list, dtype=np.float32)

    # 8. Generate the tennis court reference model
    court_ref_img, ref_points_coords = create_tennis_court_reference(scale=50, line_thickness = line_thickness)

    # 9. Prepare points for homography computation
    img_pts = []
    ref_pts_valid = []

    # We use all refined keypoints, so ref_points_coords corresponds directly
    for i in range(len(refined_keypoints_global)):
        img_pts.append(refined_keypoints_global[i])
        ref_pts_valid.append(ref_points_coords[i])

    img_pts = np.array(img_pts, dtype=np.float32)
    ref_pts_valid = np.array(ref_pts_valid, dtype=np.float32)

    # 10. Compute the homography matrix
    H = compute_homography(image_points=img_pts, ref_points=ref_pts_valid)

    # 11. Overlay the tennis court reference onto the original image
    processed_image = overlay_court(image=image_bgr, court_ref=court_ref_img, H=H)

    # 12. Draw the refined keypoints on the final image
    for i, pt in enumerate(refined_keypoints_global_list):
        x, y = pt
        x, y = int(round(x)), int(round(y))

        # Draw refined point
        cv2.circle(processed_image, (x, y), 5, (0, 0, 255), -1) # Red circle

        # Draw keypoint index
        cv2.putText(
            processed_image,
            str(i),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    return processed_image, refined_keypoints_global, H