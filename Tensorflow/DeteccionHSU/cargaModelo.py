import tensorflow as tf
import numpy 
import cv2
import matplotlib.pyplot as plt
import tensorflow_datasets as tfd
"""""codigo comentado si se quiere usar imagenes de emnist para pruebas
#funcion de normalizacion
def normalizar(images,labels):
    images=tf.cast(images,tf.float32)#convierte los valores de las imagenes a float32
    images/=255
    return images, labels
#descarga de imagenes
data,metadata=tfd.load("mnist",as_supervised=True,with_info=True)
#obetener imagenes de entrenamietno y prueba
data_train,data_test=data["train"],data["test"]
#normalizar
data_train=data_train.map(normalizar)#aplica los valores de data a la funcion normalizar
data_test=data_test.map(normalizar)
#agregar a cache
data_train=data_train.cache()
data_test=data_test.cache()
"""
#carga de modelo e imagen de prueba
model=tf.keras.models.load_model("C:\\Users\\ferna\\Documents\\vision_artificial\\Tensorflow\\DeteccionHSU\\redConvolucionalHSU.h5")
image=cv2.imread("C:\\Users\\ferna\Documents\\vision_artificial\\Tensorflow\\DeteccionHSU\\fotos\\white2.jpg",cv2.IMREAD_GRAYSCALE)
if image is None:
    print("error al cargar la imagen")
image=cv2.resize(image,(28,28))
#condicional para distinguir entre imagen en blanco o imagen con letra
if numpy.all(image>100) :
    print("blanco")
else:
    print("letra")
    letter_detected=["H","S","U"]
    #normalizar imagen 
    image=image/255.0
    image = numpy.expand_dims(image, axis=-1)  # Agregar un canal (grayscale)
    image = numpy.expand_dims(image, axis=0)  # Agregar dimensión de batch
    #mostrar imagen normalizada
    cv2.imshow("Imagen", image.squeeze())  # Eliminar dimensiones extra para mostrar la imagen
    # Espera a presionar una tecla para cerrar la imagen
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #procesar resultado e imprimirlo
    resultado=model.predict(image)
    resultado=numpy.argmax(resultado)
    resultado=letter_detected[resultado]
    print(resultado)
