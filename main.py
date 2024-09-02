import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD
import sklearn
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('altura_peso.csv')

x = df['Altura']
y = df['Peso']

print(df.head())
print("Alturas:", x.head())
print("Pesos:", y.head())

model = Sequential()

#capa densa con una entrada y una salida
model.add(Dense(units=1, input_shape=(1,), activation='linear'))

#tasa de aprendizaje de 0.0004
optimizer = SGD(learning_rate=0.0004)

#compilar el modelo
model.compile(optimizer=optimizer, loss='mse')

model.summary()

# Numero de épocas y tamaño del batch
epochs = 10000
batch_size = len(x)  #el tamaño del batch es igual al número total de datos

history = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

#Imprimir los parámetros del modelo
weights, bias = model.get_weights()
print(f"Peso (w): {weights[0][0]}")
print(f"Sesgo (b): {bias[0]}")

# Graficar el error cuadrático medio vs el número de épocas
plt.plot(history.history['loss'])
plt.title('Error cuadrático medio (ECM) durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('ECM (Loss)')
plt.show()

plt.scatter(x, y, label='Datos originales', color='blue')

#calcular la recta de regresión con los parámetros entrenados
y_pred = model.predict(x)

# Graficos
plt.plot(x, y_pred, color='red', label='Recta de regresión')

plt.title('Recta de regresión sobre los datos originales')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.show()

#prediccion: altura específica (por ejemplo, 170 cm)
altura_especifica = 170
peso_predicho = model.predict(np.array([[altura_especifica]]))
print(f"Predicción: El peso para una altura de {altura_especifica} cm es {peso_predicho[0][0]:.2f} kg")