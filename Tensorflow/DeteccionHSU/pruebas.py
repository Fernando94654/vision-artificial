import tensorflow_datasets as tfd
import cv2
import tensorflow as tf
import numpy
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
#generar imagenes en blanco
def generateWhiteImages(no_images=10000,image_size=(28,28)):
  white_images=numpy.ones((no_images, *image_size),dtype=numpy.uint8)*255
  #convertir a tensor
  white_images=tf.convert_to_tensor(white_images,dtype=tf.float32)
  white_images/=255#normalizar a [0,1]
  #crear etiquetas y pasarlas a tensor
  white_labels=numpy.full((no_images,),3,dtype=numpy.int64)
  white_labels=tf.convert_to_tensor(white_labels,dtype=tf.int64)
  return white_images,white_labels
# Cargar el dataset EMNIST con información
data, metadata = tfd.load("emnist", as_supervised=True, with_info=True)
data_train,data_test=data["train"],data["test"]

data_train=data_train.filter(filterHSU).map(change_labels)
data_train=data_train.map(normalizar)#aplica los valores de data a la funcion normalizar
#generar imagenes es blanco
white_images,white_labels=generateWhiteImages()
#generar dataset de tensorflor con imagenes blancas
white_dataset=tf.data.Dataset.from_tensor_slices((white_images,white_labels))
#concatenar imagenes a las demas
data_train=data_train.concatenate(white_dataset)
# Obtener el mapeo de etiquetas a caracteres
#label_mapping = metadata.features["label"].int2str

# Definir las letras que queremos buscar
target_letters = ['H', 'S', 'U']

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

for i, (imagen, etiqueta) in enumerate(white_dataset.take(25)):
  imagen = imagen.numpy().reshape((28,28))
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(imagen, cmap=plt.cm.binary)
  plt.xlabel(etiqueta)

plt.show()  
