import tensorflow as tf
import numpy
import cv2
import matplotlib.pyplot as plt

"""""
modelo = tf.keras.models.load_model("C:\\Users\\ferna\\Documents\\vision_artificial\\Tensorflow\\C2F.keras")
print("100 Celsius a fahrenheit:")
resultado=modelo.predict(np.array([100.0])){{{}}}
print(resultado,"fahrenheit")
"""
model=tf.keras.models.load_model("C:\\Users\\ferna\\Documents\\vision_artificial\\Tensorflow\\deteccionDeNumeros.h5")
numero=cv2.imread("C:\\Users\\ferna\Documents\\vision_artificial\\Tensorflow\\ocho.png",cv2.IMREAD_GRAYSCALE)
if numero is None:
    print("error al cargar la imagen")
numero=cv2.resize(numero,(28,28))
numero=numero/255.0
numero = numpy.expand_dims(numero, axis=-1)  # Agregar un canal (grayscale)
numero = numpy.expand_dims(numero, axis=0)  # Agregar dimensión de batch
cv2.imshow("Imagen", numero.squeeze())  # Eliminar dimensiones extra para mostrar la imagen
# Espera hasta que el usuario presione una tecla
cv2.waitKey(0)

# Cierra todas las ventanas abiertas de OpenCV
cv2.destroyAllWindows()
classes=["0","1","2","3","4","5","6","7","8","9"]

resultado=model.predict(numero)
resultado=numpy.argmax(resultado)
resultado=classes[resultado]
print(resultado)