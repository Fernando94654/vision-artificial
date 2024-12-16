import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#el as hace que al llamar funciones de tenserflow las puedas llamar como tf.funcion()
#creacion de arreglos 
celsius=np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit=np.array([-40,14,32,46,59,72,100],dtype=float)
#----------------------------------------------------------------
#creacion de capa con keras 1capa-1neurona
#capa= tf.keras.layers.Dense(units=1,input_shape=[1])#el input_shape le dice que la entrada es de una neurona
#modelo=tf.keras.Sequential([capa])
#----------------------------------------------------------------
#2capas de 3 neuronas
capa1=tf.keras.layers.Dense(units=3,input_shape=[1])
capa2=tf.keras.layers.Dense(units=3)
salida=tf.keras.layers.Dense(units=1)
modelo=tf.keras.Sequential([capa1,capa2,salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),#se da un valor de que tan rapido ira ajustando los pesos y sesgos 
    loss='mean_squared_error'#le dice que una poca cantidad de errrores grandes es peor que una gran cantidad de errores pequeños     
)
#entrenamiento
print("Comenzando entrenamiento...")
#se dan valores de entrada, esperados, no. de vueltas y no impimir todo el proceso
historial=modelo.fit(celsius,fahrenheit,epochs=100,verbose=False)
print("Modelo entrenado")
#Graficar magnitud de perdida

plt.xlabel("No. vuelta")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()

#Prediccion:
print("100 Celsius a fahrenheit:")
resultado=modelo.predict(np.array([100.0]))
print(resultado,"fahrenheit")
#modelo.save("C2F.h5")

