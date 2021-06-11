# Ejercicio 2. Cargar el archivo "TransformacionRegresionLineal.csv", a partir de la gráfica de dispersión proponer una transformación de la variable X para poder ajustar una regresión lineal.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                         "session-7-simple-linear-regression",
                         "data", "transformacion-regresion-lineal.csv")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
MB2 = pd.read_csv(inputPath, dtype='str', encoding="ISO-8859-1")
print(MB2.head())


# Función para el cálculo de los coeficientes de la regresión lineal
def RLS(x, y):
    n = len(y)
    m = (-np.dot(x, y) + float(n) * np.mean(y) * np.mean(x)) / (-np.dot(x, x) + n * (np.mean(x) ** 2.))
    b = np.mean(y) - m * np.mean(x)
    # Definición del vector de valores estimados y de residuales
    ye = []
    e = []
    for i in range(n):
        ye.extend([m * x[i] + b])
        e.extend([y[i] - ye[i]])
    return [m, b, ye, e]


# Ajuste tipo de datos
MB2["X"] = MB2["X"].astype(float)
MB2["Y"] = MB2["Y"].astype(float)

# Vectores de datos
x = MB2["X"].values.tolist()
y = MB2["Y"].values.tolist()

# Gráfica (datos orginales)
fig = plt.figure(figsize=(10., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.scatter(x, y, color="blue", s=25, label="Obs. Real")
plt.legend(loc='best')
plt.show()

# Transformación datos X
MB2["X_ajustada"] = MB2["X"] ** 2.
MB2.head()

# Ajuste regresión (línea sin transformación de datos)
[m, b, ye, e] = RLS(x, y)

# Impresión de pendiente y ordenada al origen
print("La pendiente estimada es", m)
print("La ordenda al origen estimada es", b)
# Gráfica (Regresión Lineal)
fig = plt.figure(figsize=(10., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.scatter(x, y, color="blue", s=25, label="Obs. Real")
plt.plot(x, ye, color="red", linewidth=1.5, label="Ajuste Lineal")
plt.legend(loc='best')
plt.show()

# Ajuste regresión (línea con transformación de datos)
xajustada = MB2["X_ajustada"].values.tolist()
[m, b, ye, e] = RLS(xajustada, y)
# Impresión de pendiente y ordenada al origen
print("La pendiente estimada es", m)
print("La ordenda al origen estimada es", b)
# Gráfica (Regresión Lineal)
fig = plt.figure(figsize=(10., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.scatter(x, y, color="blue", s=25, label="Obs. Real")
plt.plot(x, ye, color="red", linewidth=1.5, label="Ajuste Lineal")
plt.legend(loc='best')
plt.show()
