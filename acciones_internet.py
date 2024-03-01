#
# https://www.codificandobits.com/blog/tutorial-prediccion-de-acciones-en-la-bolsa-redes-lstm/ 
#

import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

import os

#
# Funciones auxiliares
#
def graficar_predicciones(real, prediccion):
    plt.figure()
    plt.plot(real[0:len(prediccion)],color='red', label='valores reales')
    plt.plot(prediccion, color='blue', label='predicciones')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la acción')
    plt.legend()
    plt.show()

#
# Lectura de los datos
#
file_path = os.path.abspath('archivo.csv')

##dataset = pd.read_csv('ejercicios_con_py/archivo.csv', index_col='Date', parse_dates=['Date'])

##dataset.head()

#
# Sets de entrenamiento y validación 
# La LSTM se entrenará con datos de 2016 hacia atrás. La validación se hará con datos de 2017 en adelante.
# En ambos casos sólo se usará el valor más alto de la acción para cada día
#
plt.figure()





# Cargar datos desde el archivo
datos = []
try:
    with open('datos.txt', 'r') as archivo:
        lineas = archivo.readlines()

        # Procesar cada línea para obtener los datos antes y después de la coma:
        for linea in lineas:
            partes = linea.split(',')  # Divide la línea en función de la coma
            if len(partes) == 2:  # Asegurarse de que haya dos partes (antes y después de la coma)
                dato_despues_de_coma = float(partes[1].strip())  # Obtiene la parte después de la coma y elimina espacios en blanco
                datos.append(dato_despues_de_coma)

except FileNotFoundError:
    print("El archivo no se encontró")
except IOError:
    print("Ocurrió un error al leer el archivo")

# Normalizar los datos
    
# Configuración de la señal senoidal
amplitud = 1.0
frecuencia = 0.1
muestras = 1000

# Crear valores para el eje x (tiempo)
tiempo = np.linspace(0, 200, muestras)

# Crear señal senoidal pura
senoidal_pura = amplitud * np.sin(2 * np.pi * frecuencia * tiempo)

# Agregar ruido gaussiano a la señal
ruido = np.random.normal(0, 0, muestras)
datos = senoidal_pura + ruido




maximo_valor = max(datos)
datos_normalizados = [valor / maximo_valor for valor in datos]

# Dividir los datos en entrenamiento y prueba
proporcion_datos_test = 0.8
cantidad_datos_test = int(len(datos_normalizados) * proporcion_datos_test)
datos_entrenamiento = datos_normalizados[:cantidad_datos_test]
datos_test = datos_normalizados[cantidad_datos_test:]


# Convertir datos a arrays numpy
datos_entrenamiento = np.array(datos_entrenamiento)
datos = np.array(datos)
datos_normalizados = np.array(datos_normalizados)
datos_test = np.array(datos_test)

# Crear mediciones
cantidad_mediciones_entrenamiento = len(datos_entrenamiento)
mediciones_entrenamiento = np.arange(cantidad_mediciones_entrenamiento)

set_entrenamiento = datos_entrenamiento
set_validacion = datos_test

#set_entrenamiento = dataset[:'2016'].iloc[:,1:2]
#set_validacion = dataset['2017':].iloc[:,1:2]


#plt.legend(['Entrenamiento (2006-2016)', 'Validación (2017)'])
#plt.show()

# Normalización del set de entrenamiento
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento.reshape(-1, 1))
#set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 1
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#
# Red LSTM
#
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada, return_sequences=True))
modelo.add(LSTM(units=40, input_shape=dim_entrada, return_sequences=True))
modelo.add(LSTM(units=30, input_shape=dim_entrada, return_sequences=True))
modelo.add(LSTM(units=20, input_shape=dim_entrada, return_sequences=True))
modelo.add(LSTM(units=10, input_shape=dim_entrada, return_sequences=True))



modelo.add(LSTM(units=2, input_shape=dim_entrada))

modelo.add(Dense(units=dim_salida))

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
modelo.fit(X_train,Y_train,epochs=200,batch_size=10)


#
# Validación (predicción del valor de las acciones)
#
x_test = set_entrenamiento
x_test = x_test.reshape(-1, 1)
x_test = sc.transform(x_test)


X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)


#
# Validación 2 (predicción validacion)
#
x_test2 = datos_test
x_test2 = x_test2.reshape(-1, 1)
x_test2 = sc.transform(x_test2)


X_test2 = []
for i in range(time_step,len(x_test2)):
    X_test2.append(x_test2[i-time_step:i,0])
X_test2 = np.array(X_test2)
X_test2 = np.reshape(X_test2, (X_test2.shape[0],X_test2.shape[1],1))

prediccion2 = modelo.predict(X_test2)
prediccion2 = sc.inverse_transform(prediccion2)

# Graficar resultados
graficar_predicciones(set_entrenamiento,prediccion)
graficar_predicciones(datos_test,prediccion2)