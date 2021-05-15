import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science", "session-7-simple-linear-regression",
                         "../data", "estatura-bebe.csv")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
estaturaBebeCSV = pd.read_csv(inputPath, dtype="str", encoding="ISO-8859-1")
print(estaturaBebeCSV.head())


#Ajuste de formato de columnas
estaturaBebeCSV["X"] = estaturaBebeCSV["Edad_(meses)"].astype(float)
estaturaBebeCSV["Y"] = estaturaBebeCSV["Estatura_(centimetros)"].astype(float)
print(estaturaBebeCSV.head())


x = estaturaBebeCSV["X"].values.tolist()
y = estaturaBebeCSV["Y"].values.tolist()

#Regresión lineal con la librería
from sklearn.linear_model import LinearRegression #https://scikit-learn.org/stable/
xa = np.asarray(x)
ya = np.asarray(y)
lin_reg = LinearRegression()
lin_reg.fit(xa.reshape(-1,1),ya.reshape(-1,1))
xa = np.array(x)
print(lin_reg.intercept_)
print(lin_reg.coef_)


y13=lin_reg.coef_*13+lin_reg.intercept_
print(y13)
