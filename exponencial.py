import tensorflow as tf
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