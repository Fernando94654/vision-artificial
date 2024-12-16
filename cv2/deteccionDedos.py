import cv2
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara")
    exit()

while True:
    # Leer un cuadro de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el cuadro")
        break

    # Girar el cuadro horizontalmente (opcional, para que actúe como un espejo)
    frame = cv2.flip(frame, 1)

    # Mostrar el video en tiempo real
    cv2.imshow('Video en tiempo real', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Continuar si se detecta al menos un contorno
    if len(contours) > 0:
        # Encontrar el contorno más grande (suponiendo que es la mano)
        max_contour = max(contours, key=cv2.contourArea)

        # Dibujar el contorno en el marco
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

        # Encontrar el casco convexo (convex hull)
        hull = cv2.convexHull(max_contour, returnPoints=False)

        # Encontrar los defectos convexos
        defects = cv2.convexityDefects(max_contour, hull)

        if defects is not None:
            count_fingers = 0

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Calcular las distancias
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(far) - np.array(end))

                # Calcular el ángulo
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

                # Si el ángulo es menor a 90 grados, es un dedo levantado
                if angle <= np.pi / 2:
                    count_fingers += 1
                    cv2.circle(frame, far, 4, (0, 0, 255), -1)

            # Mostrar el número de dedos levantados
            cv2.putText(frame, f'Dedos: {count_fingers}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Detección de dedos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
