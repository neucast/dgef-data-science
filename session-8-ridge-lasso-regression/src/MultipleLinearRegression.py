# Regresión lineal múltiple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                         "session-8-",
                         "data", "RegresionLinealMultiple-212.csv")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
MRLM = pd.read_csv(inputPath, dtype='str', encoding="ISO-8859-1")
MRLM["Variable_Y"] = MRLM["Variable_Y"].astype(float)
MRLM["Variable_X1"] = MRLM["Variable_X1"].astype(float)
MRLM["Variable_X2"] = MRLM["Variable_X2"].astype(float)
MRLM["Variable_X3"] = MRLM["Variable_X3"].astype(float)
MRLM.head()

# Importación de librería
from sklearn.linear_model import LinearRegression

X = np.asarray(MRLM[["Variable_X1", "Variable_X2", "Variable_X3"]])
y = np.asarray(MRLM[["Variable_Y"]])
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)

MRLM["Y_Aproximada"] = lin_reg.intercept_[0] + lin_reg.coef_[0][0] * MRLM["Variable_X1"] + \
                       lin_reg.coef_[0][1] * MRLM["Variable_X2"] + lin_reg.coef_[0][2] * MRLM["Variable_X3"]
MRLM.head()

# Gráfica entre los valores reales y los valores aproximados
fig = plt.figure(figsize=(10., 5.))
ax = fig.add_subplot(1, 1, 1)
plt.plot(range(len(MRLM["Variable_Y"].values)), MRLM["Variable_Y"].values, color="blue", linewidth=1.5,
         label="Datos originales")
plt.plot(range(len(MRLM["Y_Aproximada"].values)), MRLM["Y_Aproximada"].values, color="red", linewidth=1, ls="--",
         label="Aproximación lineal")
plt.legend(loc='best')
plt.show()

# Ejercicio 3. Agregar dos columnas a la base MRLM: 1) Columna que calcule las diferencias al cuadrado entre el valor real y el valor aproximado de la variable Y, 2) columna que calcule las diferencias del inciso anterior en valor absoluto. Imprimir el promedio de cada columna por separado.

MRLM["Dif_Cuadrado"] = (MRLM["Variable_Y"] - MRLM["Y_Aproximada"]) ** 2.
MRLM["Dif_ValorAbsoluto"] = abs(MRLM["Variable_Y"] - MRLM["Y_Aproximada"])
MRLM.head()

print("El promedio de las diferencias al cuadrado (norma L2) es igual a: ", MRLM["Dif_Cuadrado"].mean())
print("El promedio de las diferencias en valor absoluto (norma L1) es igual a: ", MRLM["Dif_ValorAbsoluto"].mean())
