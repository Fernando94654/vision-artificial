#DETECCION DE NUMEROS ESCRITOS A MANO
import tensorflow as tf
import tensorflow_datasets as tfd
import numpy
import matplotlib.pyplot as plt
import math
import cv2
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
classes=["0","1","2","3","4","5","6","7","8","9"]
#modelo
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape=(28,28,1),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),#2x2 tamaño de la matriz
    
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),#Flatten pasa de matriz 28x28 a capa de entrada de 28x28 y un canal
    tf.keras.layers.Dense(units=100,activation="relu"),#relu cambia los valores negativos a 0
    tf.keras.layers.Dense(10,activation="softmax")#sofmax convierte las salidas en probabilidades que suman 1
])
model.compile(
    optimizer="adam",#adam tipo de modelo para optimizar pesos y sesgos
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),#calcula la diferencia entre las probabilidades predichas por el modelo
    metrics=["accuracy"]# Mide qué tan bien está clasificando el modelo en términos de predicciones correctas

)
#numeros de imagenes
no_ex_train=metadata.splits["train"].num_examples
no_ex_test=metadata.splits["test"].num_examples
BATCH_SIZE=32
#sheffle hace que los datos esten de manera aleatoria y repeat hace que le conjunto de datos se repita indefinidamente
#batch agrupa los datos en el numero por lote,el entrenamiento no es orden
data_train=data_train.repeat().shuffle(no_ex_train).batch(BATCH_SIZE)
data_test=data_test.batch(BATCH_SIZE)
#entrenamiento
history=model.fit(
    data_train,
    epochs=60,
    steps_per_epoch=math.ceil(no_ex_train/BATCH_SIZE)#calcula el numero de pasos por cada epoca,math.cel redonde hacia arriba al entero mas cercano
)
model.save("redConvolucional.h5")