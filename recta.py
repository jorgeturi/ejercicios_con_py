import tensorflow as tf
import numpy as np

eje_x = np.arange(500)
eje_y = np.arange(200, 700, dtype=int)





print(eje_x)
print(eje_y)




import matplotlib.pyplot as plt
    # Definir el modelo de red neuronal
    #model = tf.keras.Sequential([
    #tf.keras.layers.Input(shape=(1,)),  # Capa de entrada con una dimensión
    # tf.keras.layers.Dense(units=1, activation='relu'),  # Capa densa con un solo nodo (la salida)
oculta1 = tf.keras.layers.Dense(units=5, input_shape=[1])
#salida = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([oculta1])


tf.keras.layers.Dense(units=1)  # Capa densa con un solo nodo (la salida)


# Compilar el modelo
model.compile(optimizer= tf.keras.optimizers.Adam(0.1), loss='mse')  # sgd: Descenso de gradiente estocástico, mse: Error cuadrático medio

# Entrenar el modelo
historial = model.fit(eje_x, eje_y, epochs=300, verbose=0)
#veo error
plt.figure(figsize=(10, 5))

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

# Evaluar el modelo en los datos de entrenamiento
loss = model.evaluate(eje_x, eje_y)
print(f'Loss en datos de entrenamiento: {loss}')

# Predecir valores para la recta
x_pred = np.arange(500)
print(oculta1.get_weights())
#print(oculta2.get_weights())
#print(salida.get_weights())
y_pred = model.predict(x_pred)

# Visualizar los resultados
plt.figure(figsize=(10, 5))

plt.scatter(eje_x, eje_y, label='Datos de entrenamiento')
plt.plot(x_pred, y_pred, color='red', label='Recta predicha')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
