import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
from collections import deque
import glob
from tqdm.auto import tqdm
from google.colab import files
from google.colab import drive
import torch.nn as nn
import torch.nn.functional as F

TARGET_W = 640
TARGET_H = 360
BATCH_SIZE = 32

class TrackNetDataset(Dataset):
    def __init__(self, sequences, input_height=360, input_width=640, is_train=False):
        self.sequences = sequences
        self.h = input_height
        self.w = input_width
        self.is_train = is_train
        self.xx, self.yy = np.meshgrid(np.arange(self.w), np.arange(self.h))

    def __len__(self):
        return len(self.sequences)

    def generate_heatmap(self, center_x, center_y, sigma=2.5):
        dist_sq = (self.xx - center_x)**2 + (self.yy - center_y)**2
        heatmap = np.exp(-dist_sq / (2 * sigma**2))
        return heatmap.astype(np.float32)

    def load_and_resize_rgb(self, path, aug_params=None):
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.w, self.h))

        # Conversión a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # APLICAR DATA AUGMENTATION
        if aug_params:
            # 1. Brillo/Contraste
            if 'alpha' in aug_params:
                # alpha: contraste (1.0 = original), beta: brillo
                img_rgb = cv2.convertScaleAbs(img_rgb, alpha=aug_params['alpha'], beta=aug_params['beta'])

            # 2. Ruido (Simular grano de cámara mala)
            if 'noise' in aug_params and aug_params['noise']:
                row, col, ch = img_rgb.shape
                mean = 0
                sigma = 15 # Intensidad del ruido
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                # Sumamos ruido y aseguramos que esté entre 0 y 255
                img_rgb = cv2.add(img_rgb, gauss.astype(np.uint8))

        # Normalizar [0, 1] y Transponer
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_norm, (2, 0, 1))

        orig_h, orig_w = img.shape[:2]

        return img_tensor

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        paths = seq['paths']

        # GENERAR PARÁMETROS ALEATORIOS (SOLO UNA VEZ POR TRÍO)
        aug_params = {}
        if self.is_train:
            # 50% de probabilidad de cambiar brillo/contraste
            if random.random() > 0.5:
                aug_params['alpha'] = random.uniform(0.7, 1.3) # Contraste +/- 30%
                aug_params['beta']  = random.uniform(-20, 20)  # Brillo +/- 20

            # 30% de probabilidad de añadir ruido
            if random.random() > 0.7:
                aug_params['noise'] = True

        temp_img = cv2.imread(paths[1])
        if temp_img is None: orig_h, orig_w = 720, 1280 # Fallback
        else: orig_h, orig_w = temp_img.shape[:2]

        t0 = self.load_and_resize_rgb(paths[0], aug_params)
        t1 = self.load_and_resize_rgb(paths[1], aug_params)
        t2 = self.load_and_resize_rgb(paths[2], aug_params)

        # Concatenar (9 canales)
        input_tensor = np.concatenate([t0, t1, t2], axis=0)

        scale_x = self.w / orig_w
        scale_y = self.h / orig_h
        target_x = seq['target_x'] * scale_x
        target_y = seq['target_y'] * scale_y

        gt_heatmap = self.generate_heatmap(target_x, target_y)
        gt_heatmap = gt_heatmap[np.newaxis, :, :]

        return torch.from_numpy(input_tensor), torch.from_numpy(gt_heatmap)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""## **Definición del Modelo: TrackNet**

"""

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class TrackNet(nn.Module):
    def __init__(self, in_ch=9, base_ch=64):
        super().__init__()

        # Encoder
        self.down1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.down3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = ConvBlock(base_ch*8 + base_ch*4, base_ch*4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = ConvBlock(base_ch*4 + base_ch*2, base_ch*2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = ConvBlock(base_ch*2 + base_ch, base_ch)

        # Salida (Heatmap)
        self.last_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x) # Guardamos para Skip Connection
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(b)
        u3 = torch.cat([u3, d3], dim=1)
        x = self.conv_up3(u3)

        u2 = self.up2(x)
        u2 = torch.cat([u2, d2], dim=1)
        x = self.conv_up2(u2)

        u1 = self.up1(x)
        u1 = torch.cat([u1, d1], dim=1)
        x = self.conv_up1(u1)

        out = self.last_conv(x)
        return out

def calculate_precision(model, loader, device, threshold=0.5):
    model.eval()
    distances = []
    detected_count = 0
    total_balls = 0


    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            # Inferencia
            preds = torch.sigmoid(model(x))

            # Pasamos a CPU
            preds_np = preds.cpu().numpy()
            y_np = y.numpy()

            for i in range(len(x)):
                # Obtener coordenadas reales (GT)
                _, _, _, gt_loc = cv2.minMaxLoc(y_np[i, 0])
                # Obtener coordenadas predichas
                min_v, max_v, min_l, pred_loc = cv2.minMaxLoc(preds_np[i, 0])

                # Solo comparamos si realmente había una pelota (max real > 0.9)
                if y_np[i, 0].max() > 0.9:
                    total_balls += 1
                    # Y si el modelo la detectó
                    if max_v > threshold:
                        detected_count += 1
                        # Distancia Euclídea
                        d = np.sqrt((gt_loc[0]-pred_loc[0])**2 + (gt_loc[1]-pred_loc[1])**2)
                        distances.append(d)

    accuracy = (detected_count / total_balls) * 100
    avg_error = np.mean(distances) if distances else 0

    print(f"\nRESULTADOS:")
    print(f"   - Tasa de Detección: {accuracy:.2f}%")
    print(f"   - Error Medio de Posición: {avg_error:.2f} píxeles")

    return distances


def get_ball_position(heatmap, threshold=0.5):
    # 1. Encontrar el valor máximo y su posición
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)

    # 2. Filtro de confianza
    # Si la red no está segura (valor < threshold), decimos que no hay pelota
    if max_val < threshold:
        return None

    # 3. Retornar coordenadas (max_loc es (x, y) en cv2)
    return max_loc
    
    
def tracknet_ball_trajectory(
    img_paths,
    model,
    ia_size=(640, 360),
    max_dist_pixels=200,
    max_gap_interpolation=4
):

    first_img = cv2.imread(img_paths[0])
    H, W = first_img.shape[:2]
    device = next(model.parameters()).device

    raw_detections = []
    frame_queue = deque(maxlen=3)

    print("Analizando trayectoria")
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            raw_detections.append(None)
            continue

        frame_queue.append(img)

        if len(frame_queue) == 3:
            imgs_ia = [np.transpose(cv2.resize(f, ia_size).astype(np.float32)/255.0, (2,0,1)) for f in frame_queue]
            inp = torch.from_numpy(np.concatenate(imgs_ia, 0)).unsqueeze(0).to(device)

            with torch.no_grad():
                heatmap = torch.sigmoid(model(inp))[0, 0].cpu().numpy()

            min_v, max_v, min_l, max_l = cv2.minMaxLoc(heatmap)

            if max_v > 0.5:
                scale_x, scale_y = W / ia_size[0], H / ia_size[1]
                raw_detections.append((int(max_l[0]*scale_x), int(max_l[1]*scale_y)))
            else:
                raw_detections.append(None)
        else:
            raw_detections.append(None)

    raw_detections.pop(0)
    raw_detections.append(None)

    clean_path = raw_detections.copy()
    for i in range(1, len(clean_path) - 1):
        if clean_path[i] is not None:
            if clean_path[i-1] is None and clean_path[i+1] is None:
                clean_path[i] = None

    # FILTRO DE DISTANCIA E INTERPOLACIÓN
    processed_path = [None] * len(clean_path)
    valid_indices = [i for i, x in enumerate(clean_path) if x is not None]

    if valid_indices:
        for k in range(len(valid_indices) - 1):
            idx_curr = valid_indices[k]
            idx_next = valid_indices[k+1]

            pos_curr = np.array(clean_path[idx_curr])
            pos_next = np.array(clean_path[idx_next])

            gap = idx_next - idx_curr
            dist = np.linalg.norm(pos_next - pos_curr)

            # Solo unimos si la distancia es lógica
            if gap == 1:
                if dist <= max_dist_pixels:
                    processed_path[idx_curr] = tuple(pos_curr.astype(int))
                    processed_path[idx_next] = tuple(pos_next.astype(int))
            elif gap <= max_gap_interpolation:
                # Interpolación si el salto total no es una locura
                if dist <= (max_dist_pixels * gap):
                    for i in range(gap + 1):
                        interp_pos = pos_curr + (pos_next - pos_curr) * (i / gap)
                        processed_path[idx_curr + i] = tuple(interp_pos.astype(int))

    # SUAVIZADO (MOVING AVERAGE)
    # Promediamos la posición actual con la anterior y la siguiente
    smooth_path = processed_path.copy()
    for i in range(1, len(processed_path) - 1):
        p_prev = processed_path[i-1]
        p_curr = processed_path[i]
        p_next = processed_path[i+1]

        if p_prev and p_curr and p_next:
            # Promedio de coordenadas X e Y
            avg_x = int((p_prev[0] + p_curr[0] + p_next[0]) / 3)
            avg_y = int((p_prev[1] + p_curr[1] + p_next[1]) / 3)
            smooth_path[i] = (avg_x, avg_y)
    return smooth_path



def draw_ball_and_trail(img, ball_pos, trail, max_trail=8):
    """
    trail: deque compartida entre frames
    ball_pos: (x,y) o None
    """
    trail.append(ball_pos)

    # Estela
    for j in range(1, len(trail)):
        if trail[j-1] and trail[j]:
            thick = int(np.sqrt(64 / (len(trail) - j + 1)) * 2) + 1
            cv2.line(img, trail[j-1], trail[j], (0, 0, 255), thick)

    # Pelota actual
    if trail[-1]:
        cv2.circle(img, trail[-1], 6, (0, 255, 255), -1)
        cv2.circle(img, trail[-1], 7, (0, 0, 0), 1)

    return img

