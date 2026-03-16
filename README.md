# Tennis Tracker VC  
Sistema de Análisis Automático de Partidos de Tenis mediante Visión por Computador

---

## Descripción

Este proyecto implementa un sistema completo para el análisis automático de partidos de tenis a partir de vídeo broadcast.

El sistema es capaz de:

- Detectar la pista de tenis y sus keypoints geométricos
- Calcular una homografía robusta
- Detectar y seguir a los jugadores
- Seguir la pelota mediante Deep Learning (TrackNet)
- Proyectar posiciones a un minimapa cenital
- Calcular métricas cinéticas básicas
- Generar un vídeo final integrado

El enfoque combina redes neuronales convolucionales, técnicas clásicas como Hough Lines y geometría proyectiva.

---

# Pipeline General

![Pipeline General](assets/pipeline_overview.png)  
<!-- INSERTAR IMAGEN DEL ROADMAP GENERAL -->

El flujo completo del sistema es:

1. Detección de pista y keypoints  
2. Tracking de jugadores (YOLO + filtrado geométrico)  
3. Homografía inversa y proyección al plano real  
4. Seguimiento de la pelota (TrackNet)  
5. Integración final y generación de vídeo con métricas  

---

# Módulo 1: Detección de la Pista

## Enfoque

Se utiliza MobileNetV3-Small preentrenado adaptado a regresión para predecir 14 keypoints estratégicos de la pista.

![Keypoints Detectados](assets/keypoints_raw.png)  
<!-- INSERTAR IMAGEN KEYPOINTS SIN REFINAR -->

## Refinamiento Geométrico

Debido a pequeñas imprecisiones, se aplica un postprocesado que incluye:

- Extracción de crops por keypoint  
- K-Means para simplificación de color  
- Umbralización  
- Dilatación  
- Afinado Zhang-Suen  
- Hough Lines  
- Selección de intersección geométricamente estable  

![Refinamiento](assets/keypoints_refined.png)  
<!-- INSERTAR IMAGEN ANTES/DESPUÉS -->

Resultado: keypoints alineados con intersecciones reales.

---

# Módulo 2: Homografía y Proyección

A partir de los keypoints refinados se calcula una homografía robusta mediante RANSAC.

Esto permite:

- Proyectar líneas reales sobre la imagen  
- Eliminar efecto de perspectiva  
- Transformar coordenadas imagen → plano real  

![Homografía](assets/homography_overlay.png)  
<!-- INSERTAR IMAGEN DE PROYECCIÓN -->

---

# Módulo 3: Tracking de Jugadores

## Detección

Se utiliza YOLO para detectar personas.

![YOLO Detection](assets/yolo_players.png)  
<!-- INSERTAR IMAGEN YOLO -->

## Selección Geométrica

Para seleccionar solo los jugadores:

- Se proyectan los pies al plano real  
- Se utilizan keypoints virtuales  
- Se asigna un jugador por cada mitad  

Esto evita errores por perspectiva o detecciones espurias.

---

# Módulo 4: Seguimiento de Pelota (TrackNet)

Se implementa TrackNet, arquitectura tipo U-Net que:

- Procesa secuencias de 3 frames  
- Predice mapas de calor  
- Usa Focal Loss para mitigar desbalance  

![Heatmap TrackNet](assets/tracknet_heatmap.png)  
<!-- INSERTAR HEATMAP -->

## Postprocesado

Para mejorar estabilidad:

- Alineación espacial  
- Filtrado lógico  
- Interpolación temporal  
- Suavizado de trayectoria  

Resultado: trayectorias más estables y coherentes.

![Trayectoria Suavizada](assets/ball_trajectory.png)  
<!-- INSERTAR IMAGEN DE TRAYECTORIA -->

---

# Minimapa y Visualización

Se construye un minimapa cenital con:

- Margen extendido  
- Overlay transparente  
- Proyección de jugadores  
- Proyección de pelota  

![Minimapa](assets/minimap.png)  
<!-- INSERTAR IMAGEN DEL MINIMAPA -->

---

# Demo Final

El sistema integra todos los módulos para generar un vídeo final con:

- Detección de pista  
- Tracking de jugadores  
- Seguimiento de pelota  
- Minimapa dinámico  
- Métricas cinéticas  

Vídeo Demo:

[Ver demo completa](INSERTAR_LINK_A_VIDEO)


#Tecnologías Utilizadas

-Python
-PyTorch
-OpenCV
-YOLO
-MobileNetV3
-TrackNet
-RANSAC
-Hough Transform

#Resultados

-Error medio keypoints: ~7–8 píxeles
-TrackNet: convergencia estable
-Reducción significativa de falsos positivos tras postprocesado
-Proyección geométrica coherente en plano real


#Autores:
Marino Fernandez Pérez, Pau Bover Femenias , Francesc Oliver Catany, Gabriele Ruggeri
Proyecto desarrollado en el marco de Visión por Computador – Universidad de Granada (UGR).

