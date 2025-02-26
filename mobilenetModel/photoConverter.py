
#artificial vision model using the google mobilenet that uses imagenet dataset to detect H-S-U letters
import tensorflow_hub as hub
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
input_path="C:\\Users\\ferna\\Documents\\vision_artificial\\fotos\\dataset_HSUGYR\\none"
outputh_path="C:\\Users\\ferna\\Documents\\vision_artificial\\fotos\\grayscaleDataset\\none"
os.makedirs(outputh_path,exist_ok=True)#crear carpeta si no ha sido creada

for archivo in os.listdir(input_path):#recorrer archivos
    if archivo.endswith(".jpg") or archivo.endswith(".png"):#si es imagen
        img=Image.open(os.path.join(input_path,archivo))#abrir imagen usando PIL
        img_gray=img.convert("L")#convertir a grises
        img_gray.save(os.path.join(outputh_path,f"graycale_{archivo}"))#guardar
print("completado")