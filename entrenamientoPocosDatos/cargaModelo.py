import tensorflow as tf
import numpy 
import cv2
import matplotlib.pyplot as plt
import tensorflow_datasets as tfd
from keras.utils import custom_object_scope
import tensorflow_hub as hub

with custom_object_scope({'KerasLayer': hub.KerasLayer}):
    model = tf.keras.models.load_model("C:\\Users\\ferna\\Documents\\vision_artificial\\entrenamientoPocosDatos\\HSU_detection_mobilenet.h5")
image=cv2.imread("C:\\Users\\ferna\\Documents\\vision_artificial\\fotos\\H\\H5.jpeg")
if image is None:
    print("error al cargar la imagen")

image = numpy.array(image).astype(float)/255
image = cv2.resize(image, (224,224))
cv2.imshow("Imagen", image.squeeze())  # Eliminar dimensiones extra para mostrar la imagen
# Espera a presionar una tecla para cerrar la imagen
cv2.waitKey(0)
cv2.destroyAllWindows()
resultado = model.predict(image.reshape(-1, 224, 224, 3))
resultado=numpy.argmax(resultado)
print(resultado)
