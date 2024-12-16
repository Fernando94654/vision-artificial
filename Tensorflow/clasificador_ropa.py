import tensorflow as tf
import tensorflow_datasets as tfd
import numpy
import matplotlib.pyplot as plt
import math
import cv2
#normalizar los datos (pasar de 255 a 0-1)
def normalizar(images,labels):
    images=tf.cast(images,tf.float32)#Convierte los valores de enteros a flotantes 
    images/= 255 #se pasa de 255 a 0-1, divide cada valor entre 255
    return images, labels

data, metadata=tfd.load("fashion_mnist",as_supervised=True,with_info=True)
#as_supervised=True: Devuelve las imágenes y etiquetas en formato (input, label), útil para entrenamiento supervisado.
#with_info=True: Devuelve también metadatos sobre el conjunto de datos.
#data: Contiene los datos divididos en subconjuntos (train y test).
#metadata: Contiene información adicional sobre el conjunto de datos, como las etiquetas de las clases.
#print(metadata)
data_train, data_test=data["train"],data["test"]
class_names=metadata.features["label"].names
print(class_names)
#normalizar los datos de entrenamiento y pruebas con la funcion
data_train=data_train.map(normalizar)
data_test=data_test.map(normalizar)
#agregar a cache(usar memoria en lugar de disco, entrenamiento mas rapido)
data_train=data_train.cache()
data_test=data_test.cache()
#mostrar una imagen de los datos de pruebas, solo una
"""
for image,label in data_train.take(1):
    break
image=image.numpy().reshape((28,28))#redimensionar, cosas de tensores, se vera despues
#dibujar 
plt.figure()
plt.imshow(image,cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#mostrar y etiquetar varias imagenes
for i, (image,label) in enumerate(data_train.take(25)):
    image=image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()
"""
#crear modelo
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),#pasa la matriz 28x28 a una entrada de 784 neuronas
    tf.keras.layers.Dense(50,activation=tf.nn.relu),#creacion de 2 capas ocultas
    tf.keras.layers.Dense(50,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)#capa de salida,sofmax para salida de redes de clasificacion
])
#compilar modelo
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
num_ex_train=metadata.splits["train"].num_examples
num_ex_test=metadata.splits["test"].num_examples
#entrenamiento por lotes
LOT_SIZE=32
data_train=data_train.repeat().shuffle(num_ex_train).batch(LOT_SIZE)#vueltas aleatorias
data_test=data_test.batch(LOT_SIZE)
#entrenar
history=model.fit(data_train,epochs=5,steps_per_epoch=math.ceil(num_ex_train/LOT_SIZE))
"""""
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(history.history["loss"])
plt.show()
"""""
shoe_image=cv2.imread("C:\\Users\\ferna\Documents\\vision_artificial\\Tensorflow\\zap.png",cv2.IMREAD_GRAYSCALE)
if shoe_image is None:
    print("error al cargar la imagen")
shoe_image=cv2.resize(shoe_image,(28,28))
shoe_image=shoe_image/255.0
shoe_image = numpy.expand_dims(shoe_image, axis=-1)  # Agregar un canal (grayscale)
shoe_image = numpy.expand_dims(shoe_image, axis=0)  # Agregar dimensión de batch
#shoe_image=numpy.array([shoe_image])
result=model.predict(shoe_image)
result=numpy.argmax(result)
result=class_names[result-1]
print(result)