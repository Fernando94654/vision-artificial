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
def filterHSU(images,labels):
    return tf.reduce_any(tf.equal(labels,[17,28,30]))
def change_labels(image,label):
    etiqueta1 = tf.constant(17, dtype=tf.int64) 
    etiqueta2 = tf.constant(0, dtype=tf.int64) 
    label=tf.where(label==etiqueta1,etiqueta2,label)
    etiqueta3 = tf.constant(28, dtype=tf.int64) 
    etiqueta4 = tf.constant(1, dtype=tf.int64) 
    label=tf.where(label==etiqueta3,etiqueta4,label)
    etiqueta5 = tf.constant(30, dtype=tf.int64) 
    etiqueta6 = tf.constant(2, dtype=tf.int64) 
    label=tf.where(label==etiqueta5,etiqueta6,label)
    return image,label
#descarga de imagenes
data,metadata=tfd.load("emnist",as_supervised=True,with_info=True)
#obetener imagenes de entrenamietno y prueba
data_train,data_test=data["train"],data["test"]
#Filtrar H-S-U
data_train=data_train.filter(filterHSU).map(change_labels)
data_test=data_test.filter(filterHSU).map(change_labels)
#normalizar
data_train=data_train.map(normalizar)#aplica los valores de data a la funcion normalizar
data_test=data_test.map(normalizar)
#agregar a cache
data_train=data_train.cache()
data_test=data_test.cache()
#modelo
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), input_shape=(28,28,1),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),#2x2 tamaño de la matriz
    
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),#Flatten pasa de matriz 28x28 a capa de entrada de 28x28 y un canal
    tf.keras.layers.Dense(units=100,activation="relu"),#relu cambia los valores negativos a 0
    tf.keras.layers.Dense(3 ,activation="softmax")#sofmax convierte las salidas en probabilidades que suman 1
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
    epochs=20,
    steps_per_epoch=math.ceil(no_ex_train/BATCH_SIZE)#calcula el numero de pasos por cada epoca,math.cel redonde hacia arriba al entero mas cercano
)
model.save("redConvolucionalHSU.h5")