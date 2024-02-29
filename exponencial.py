"""import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Crear datos de entrada y salida
eje_x = np.linspace(0, 3, 500)
eje_y = np.exp(eje_x)

# Definir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Capa de entrada con una dimensión
    tf.keras.layers.Dense(units=2, activation='sigmoid'),  # Capa densa con activación lineal
    tf.keras.layers.Dense(units=2, activation='sigmoid'),  # Capa densa con activación lineal
    tf.keras.layers.Dense(units=30, activation='sigmoid'),  # Capa densa con activación lineal
    tf.keras.layers.Dense(units=2, activation='sigmoid'),  # Capa densa con activación lineal
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # Capa densa con activación lineal
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')  

# Entrenar el modelo
historial = model.fit(eje_x, eje_y, epochs=1000, verbose=1)

# Visualizar la pérdida a lo largo de las épocas
plt.figure(figsize=(10, 5))
plt.xlabel("# Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

# Evaluar el modelo en los datos de entrenamiento
loss = model.evaluate(eje_x, eje_y)
print(f'Pérdida en datos de entrenamiento: {loss}')

# Predecir valores para la recta
x_pred = np.linspace(0, 3, 500)
y_pred = model.predict(x_pred)

# Visualizar los resultados
plt.figure(figsize=(10, 5))
plt.scatter(eje_x, eje_y, label='Datos de entrenamiento')
plt.plot(x_pred, y_pred, color='red', label='Recta predicha')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

"""




"""
##########
######### DADA POR CHAT
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo (exponencial)
np.random.seed(0)
x_train = np.sort(5 * np.random.rand(100))
y_train = np.exp(x_train) + 0.1 * np.random.randn(100)

# Definir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(units=1)  # Capa de salida sin función de activación para regresión
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(x_train, y_train, epochs=1000, verbose=0)

# Generar datos para hacer predicciones
x_test = np.linspace(0, 5, 100)
y_pred = model.predict(x_test)

# Visualizar resultados
plt.scatter(x_train, y_train, label='Datos de entrenamiento')
plt.plot(x_test, y_pred, color='red', label='Predicciones')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

"""

#####
##### CON MEMORIA

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo (exponencial)
np.random.seed(0)
x_train = np.sort(5 * np.random.rand(100))
y_train = np.exp(x_train) + 0.1 * np.random.randn(100)

# Reshape para que sea compatible con la entrada de LSTM
x_train = x_train.reshape(-1, 1, 1)

# Definir el modelo con una capa LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=10, activation='relu', input_shape=(1, 1)),
    tf.keras.layers.Dense(units=1)  # Capa de salida sin función de activación para regresión
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(x_train, y_train, epochs=1000, verbose=0)

# Generar datos para hacer predicciones
x_test = np.linspace(0, 5, 100).reshape(-1, 1, 1)
y_pred = model.predict(x_test)

# Deshacer el reshape para visualización
x_test = x_test.reshape(-1)
y_pred = y_pred.reshape(-1)

# Visualizar resultados
plt.scatter(x_train.reshape(-1), y_train, label='Datos de entrenamiento')
plt.plot(x_test, y_pred, color='red', label='Predicciones')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()