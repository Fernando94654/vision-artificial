import tensorflow_datasets as tfd
import tensorflow as tf
from collections import Counter
import numpy
# Cargar el dataset EMNIST
data, metadata = tfd.load("emnist", as_supervised=True, with_info=True)
data_train, data_test = data["train"], data["test"]

# Función para normalizar
def normalizar(images, labels):
    images = tf.cast(images, tf.float32)  # Convierte los valores de las imágenes a float32
    images /= 255  # Normaliza a [0, 1]
    return images, labels

# Función para filtrar las etiquetas de interés (17, 28, 30)
def filterHSU(images, labels):
    return tf.reduce_any(tf.equal(labels, [17, 28, 30]))

# Cambiar las etiquetas
def change_labels(image, label):
    etiqueta1 = tf.constant(17, dtype=tf.int64) 
    etiqueta2 = tf.constant(0, dtype=tf.int64) 
    label = tf.where(label == etiqueta1, etiqueta2, label)
    
    etiqueta3 = tf.constant(28, dtype=tf.int64) 
    etiqueta4 = tf.constant(1, dtype=tf.int64) 
    label = tf.where(label == etiqueta3, etiqueta4, label)
    
    etiqueta5 = tf.constant(30, dtype=tf.int64) 
    etiqueta6 = tf.constant(2, dtype=tf.int64) 
    label = tf.where(label == etiqueta5, etiqueta6, label)
    
    return image, label
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
# Filtrar y mapear las funciones
data_train = data_train.filter(filterHSU).map(change_labels).map(normalizar)
#generar imagenes es blanco
white_images,white_labels=generateWhiteImages()
#generar dataset de tensorflor con imagenes blancas
white_dataset=tf.data.Dataset.from_tensor_slices((white_images,white_labels))
#concatenar imagenes a las demas
data_train=data_train.concatenate(white_dataset)

# Contar las imágenes por clase en los datos filtrados
def count_classes(dataset):
    class_counter = Counter()  # Usamos un Counter para contar las clases
    for _, label in dataset:
        class_counter[label.numpy()] += 1  # Incrementar el contador para cada etiqueta
    
    return class_counter

# Llamar a la función para contar las clases en el conjunto de datos
class_counts = count_classes(data_train)

# Mostrar el conteo de imágenes por clase
print(f"Conteo de imágenes por clase: {class_counts}")
#Conteo de imágenes por clase: Counter({np.int64(1): 20764, np.int64(2): 12602, np.int64(0): 3152})