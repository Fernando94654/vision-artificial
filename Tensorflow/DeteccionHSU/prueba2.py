import tensorflow as tf

# Verificar si TensorFlow detecta la GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("TensorFlow detectó las siguientes GPUs:")
    for gpu in gpus:
        print(f"- {gpu.name}")
else:
    print("No se detectaron GPUs.")