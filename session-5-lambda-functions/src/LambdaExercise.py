import os
import pandas as pd

findString = lambda x, y: x.find(y)

if findString("Hola", "b") >= 0:
    print("Se encuentra la cadena.")
else:
    print("No se encuentra la cadena.")

parImpar = lambda x: True if x % 2 == 0 else False

print(parImpar(2))

print(parImpar(3))

# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science", "session-5-lambda-functions",
                         "data", "capitalizacion-r-computo.csv")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
capitalizacionCSV = pd.read_csv(inputPath, dtype="str", encoding="ISO-8859-1")

# Converts to dataframe.
capitalizacionDF = pd.DataFrame(capitalizacionCSV)

# Dataframe size.
print("Original dataframe size:", capitalizacionDF.shape)

# Sets the type of each column of the data frame.
capitalizacionDF["Fecha"] = capitalizacionDF["Fecha"].astype(str)
capitalizacionDF["Institucion"] = capitalizacionDF["Institucion"].astype(str)
capitalizacionDF["ICAP"] = capitalizacionDF["ICAP"].astype(float)
capitalizacionDF["Capital Neto reg. vig. dic 2012"] = capitalizacionDF["ICAP"].astype(float)
capitalizacionDF["Capital Neto Vigente"] = capitalizacionDF["ICAP"].astype(float)
capitalizacionDF["Operaciones Sujetas a Riesgo"] = capitalizacionDF["ICAP"].astype(float)
capitalizacionDF["RC Total"] = capitalizacionDF["ICAP"].astype(float)
capitalizacionDF["RC x riesgo de Credito"] = capitalizacionDF["ICAP"].astype(float)
capitalizacionDF["RC x riesgo de Mercado"] = capitalizacionDF["ICAP"].astype(float)
capitalizacionDF["RC x riesgo Operacional"] = capitalizacionDF["ICAP"].astype(float)
capitalizacionDF["RC x riesgo de Filiales del Exterior"] = capitalizacionDF["ICAP"].astype(float)

# print(capitalizacionDF)

# Assigns the data frame indexes.
# capitalizacionDF.set_index(["Institucion", "Fecha"], inplace=True)

# print(capitalizacionDF.head(10))

# Sorts the data frame by its indexes.
# capitalizacionDF.sort_values(["Institucion", "Fecha"], ascending=True, inplace=True)

print(capitalizacionDF.head(10))

print(pd.pivot_table(capitalizacionDF, columns="Fecha", values=["Capital Neto Vigente"], index="Institucion",
                     aggfunc="sum", fill_value=0, margins=True, margins_name="Total Sistema"))

capitalizacionDF["RC xr C monto"] = capitalizacionDF["RC x riesgo de Credito"] * capitalizacionDF["RC Total"] / 100
capitalizacionDF["RC xr M monto"] = capitalizacionDF["RC x riesgo de Mercado"] * capitalizacionDF["RC Total"] / 100
capitalizacionDF["RC xr O monto"] = capitalizacionDF["RC x riesgo Operacional"] * capitalizacionDF["RC Total"] / 100
capitalizacionDF["RC xr F monto"] = capitalizacionDF["RC x riesgo de Filiales del Exterior"] * capitalizacionDF[
    "RC Total"] / 100

print(pd.pivot_table(capitalizacionDF, index="Institucion",
                     values=["RC xr C monto", "RC xr M monto", "RC xr O monto", "RC xr F monto"], aggfunc="max",
                     fill_value=0, margins=True, margins_name="Total Sistema"))
