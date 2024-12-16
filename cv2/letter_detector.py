import cv2
import pytesseract
# Configura la ubicación de Tesseract si no está en el PATH (solo en Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'  # Cambia la ruta si es necesario

# Cargar la imagen
image = cv2.imread('h.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un umbral (opcional, mejora la detección)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
text = pytesseract.image_to_string(image)
print(text)



