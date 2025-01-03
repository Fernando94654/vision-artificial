import tensorflow_hub as hub
import tensorflow as tf
#Aumento de datos con ImageData Generator
#import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
#crear dataset
datagen=ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=.25,
    height_shift_range=.25,
    shear_range=0.15,
    zoom_range=[0.5,1.5],
    validation_split=0.2#20% para pruebas
)
#generadores para sets de entrenamiento y pruebas
data_gen_train=datagen.flow_from_directory("C:\\Users\\ferna\\Documents\\vision_artificial\\fotos\dataset_HSU",target_size=(224,224),
                                           batch_size=32,shuffle=True,subset="training")
data_gen_test=datagen.flow_from_directory("C:\\Users\\ferna\\Documents\\vision_artificial\\fotos\dataset_HSU",target_size=(224,224),
                                           batch_size=32,shuffle=True,subset="validation")
datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=.25,
    height_shift_range=.25,
    shear_range=0.15,
    zoom_range=[0.5,1.5],
    validation_split=0.2#20% para pruebas
)

mobilenet_url="https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/tf2-preview-feature-vector/4"
mobilenetv2 = hub.KerasLayer(mobilenet_url, input_shape=(224,224,3))
mobilenetv2.trainable = False
model = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
#entrenar model
Epochs=50
history=model.fit(
    data_gen_train,epochs=Epochs,batch_size=32,
    validation_data=data_gen_test
)
model.save("HSU_detection_mobilenet.h5")
