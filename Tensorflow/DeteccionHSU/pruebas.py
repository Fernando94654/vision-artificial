import tensorflow_datasets as tfd
import cv2
import tensorflow as tf
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
# Cargar el dataset EMNIST con información
data, metadata = tfd.load("emnist", as_supervised=True, with_info=True)
data_train,data_test=data["train"],data["test"]

data_train=data_train.filter(filterHSU).map(change_labels)
data_train=data_train.map(normalizar)#aplica los valores de data a la funcion normalizar

# Obtener el mapeo de etiquetas a caracteres
#label_mapping = metadata.features["label"].int2str

# Definir las letras que queremos buscar
target_letters = ['H', 'S', 'U']

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

for i, (imagen, etiqueta) in enumerate(data_train.take(25)):
  imagen = imagen.numpy().reshape((28,28))
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(imagen, cmap=plt.cm.binary)
  plt.xlabel(etiqueta)

plt.show()  
