import cv2
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'  # Cambia la ruta si es necesario

#MOSTRAR IMAGEN EN PLOT
image=cv2.imread('abe.png')
"""1
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off')
plt.show()
"""
#CONVERTIR A ESCALA DE GRISES
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#MOSTRAR EN ESCALA DE GRISES
"""
gray_image_rgb=cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)
plt.imshow(gray_image_rgb)
plt.axis('off')
plt.show()
"""

#se usa la funcion Canny para la deteccion de bordes
#deteccion de bordes
#cambios bruscos de intencidad
edges=cv2.Canny(gray_image, threshold1=100,threshold2=200)
#threshold1 y 2 son umbrales para la deteccion el 2 sera tomado como borde y el 1 sera considerado
"""
#Mostrar bordes
plt.imshow(edges, cmap='gray')#se indica que se muestre edges en escala de grises
plt.axis('off')
plt.show()
"""
#DETECCION DE CONTORNOS
#los contornos son algo que estructura a los bordes de manera mas uniforme y continua
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#se pone , _ porque la funcion devuelve 2 valores pero solo quiero la de los contornos
#cv2.RETR_EXTERNAL: indica a OpenCV que solo debe encontrar los contornos exteriores.
#cv2.CHAIN_APPROX_SIMPLE: le dice a OpenCV que use una aproximación simple de los contornos 
#Dibujar bordes sobres la imagen original
image_contours=gray_image.copy()
cv2.drawContours(image_contours, contours,-1,(0,255,0),2)#nums:todos los bordes, verde, ancho de linea
#Mostrar imagen
imagen_rgb_contours=cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB)
plt.imshow(imagen_rgb_contours)
plt.axis('off')
plt.show()
text=pytesseract.image_to_string(imagen_rgb_contours)
print("texto detectado:", text)
#
