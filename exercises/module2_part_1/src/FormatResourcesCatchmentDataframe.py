import os
import pandas as pd

"""
Using the file "captacion-de-recursos-bm.csv", generate a table in pandas that contains the ordering as 
well as the rows and columns shown in the image "captacion-de-recursos-bm.jpg". 
"""

# Sets Pandas options.
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science", "exercises",
                         "module2_part_1", "data", "captacion-de-recursos-bm.csv")

# outputPath to the HTML result file.
outputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science", "exercises",
                          "module2_part_1", "output", "reporte-captacion-de-recursos-bm.html")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
resourcesCatchmentCSV = pd.read_csv(inputPath, dtype="str", encoding="ISO-8859-1")

# Converts to dataframe and adds grand total.
vistaImporteTotalSeries = pd.Series(["28/02/21", "Vista", "Importe", "TOT. SISTEMA BANCA MÚLTIPLE", 0.0])
plazoImporteTotalSeries = pd.Series(["28/02/21", "Plazo", "Importe", "TOT. SISTEMA BANCA MÚLTIPLE", 0.0])
bancarioImporteTotalSeries = pd.Series(["28/02/21", "Bancario 1_/", "Importe", "TOT. SISTEMA BANCA MÚLTIPLE",
                                        0.0])
interbancarioImporteTotalSeries = pd.Series(["28/02/21", "Interbancario 2_/", "Importe", "TOT. SISTEMA BANCA MÚLTIPLE",
                                             0.0])
otrosRecursosImporteTotalSeries = pd.Series(["28/02/21", "Otros recursos", "Importe", "TOT. SISTEMA BANCA MÚLTIPLE",
                                             0.0])
totalImporteTotalSeries = pd.Series(["28/02/21", "Total 3_/", "Importe", "TOT. SISTEMA BANCA MÚLTIPLE",
                                     0.0])

vistaCostoTotalSeries = pd.Series(["28/02/21", "Vista", "Costo", "TOT. SISTEMA BANCA MÚLTIPLE", 0.0])
plazoCostoTotalSeries = pd.Series(["28/02/21", "Plazo", "Costo", "TOT. SISTEMA BANCA MÚLTIPLE", 0.0])
bancarioCostoTotalSeries = pd.Series(["28/02/21", "Bancario 1_/", "Costo", "TOT. SISTEMA BANCA MÚLTIPLE",
                                      0.0])
interbancarioCostoTotalSeries = pd.Series(["28/02/21", "Interbancario 2_/", "Costo", "TOT. SISTEMA BANCA MÚLTIPLE",
                                           0.0])
otrosRecursosCostoTotalSeries = pd.Series(["28/02/21", "Otros recursos", "Costo", "TOT. SISTEMA BANCA MÚLTIPLE",
                                           0.0])
totalCostoTotalSeries = pd.Series(["28/02/21", "Total 3_/", "Costo", "TOT. SISTEMA BANCA MÚLTIPLE", 0.0])

grandTotalDF = pd.DataFrame(
    [list(vistaImporteTotalSeries), list(plazoImporteTotalSeries), list(bancarioImporteTotalSeries),
     list(interbancarioImporteTotalSeries), list(otrosRecursosImporteTotalSeries), list(totalImporteTotalSeries),
     list(vistaCostoTotalSeries), list(plazoCostoTotalSeries), list(bancarioCostoTotalSeries),
     list(interbancarioCostoTotalSeries), list(otrosRecursosCostoTotalSeries), (totalCostoTotalSeries)],
    columns=["Fecha", "Concepto", "Metrica", "Institucion", "Valor"])

resourcesCatchmentDF = pd.DataFrame(resourcesCatchmentCSV).append(grandTotalDF, ignore_index=False)

# Dataframe size.
print("Original dataframe size:", resourcesCatchmentDF.shape)

# Sets the type of each column of the data frame.
resourcesCatchmentDF["Fecha"] = pd.to_datetime(resourcesCatchmentDF["Fecha"], dayfirst=True)
resourcesCatchmentDF["Concepto"] = resourcesCatchmentDF["Concepto"].astype(str)
resourcesCatchmentDF["Metrica"] = resourcesCatchmentDF["Metrica"].astype(str)
resourcesCatchmentDF["Institucion"] = resourcesCatchmentDF["Institucion"].astype(str)
resourcesCatchmentDF["Valor"] = resourcesCatchmentDF["Valor"].astype(float)

# print(resourcesCatchmentDF)

# Assigns the data frame indexes.
resourcesCatchmentDF.set_index(["Fecha", "Institucion"], inplace=True)

# Sorts the data frame by its indexes.
resourcesCatchmentDF.sort_values(["Fecha", "Institucion"], ascending=True, inplace=True)

# print(resourcesCatchmentDF.head(10))

# Gets the pivot table.
pivotTable = pd.pivot_table(resourcesCatchmentDF, columns=["Metrica", "Concepto"], values=["Valor"],
                            index=["Fecha", "Institucion"])

# print(pivotTable.head(10))

# For amount.
pivotTableAmountDF = pd.DataFrame(pivotTable.iloc[:, pivotTable.columns.get_level_values(1) == "Importe"])

# print(pivotTableAmountDF.columns)

pivotTableAmountDF[("Valor", "Importe", "Total 3_/")] = pivotTableAmountDF[("Valor", "Importe", "Vista")] + \
                                                        pivotTableAmountDF[("Valor", "Importe", "Plazo")] + \
                                                        pivotTableAmountDF[("Valor", "Importe", "Bancario 1_/")] + \
                                                        pivotTableAmountDF[("Valor", "Importe", "Interbancario 2_/")] + \
                                                        pivotTableAmountDF[("Valor", "Importe", "Otros recursos")]

pivotTableAmountCol1 = pivotTableAmountDF.iloc[:, pivotTableAmountDF.columns.get_level_values(2) == "Vista"]
pivotTableAmountCol2 = pivotTableAmountDF.iloc[:, pivotTableAmountDF.columns.get_level_values(2) == "Plazo"]
pivotTableAmountCol3 = pivotTableAmountDF.iloc[:, pivotTableAmountDF.columns.get_level_values(2) == "Bancario 1_/"]
pivotTableAmountCol4 = pivotTableAmountDF.iloc[:, pivotTableAmountDF.columns.get_level_values(2) == "Interbancario 2_/"]
pivotTableAmountCol5 = pivotTableAmountDF.iloc[:, pivotTableAmountDF.columns.get_level_values(2) == "Otros recursos"]
pivotTableAmountCol6 = pivotTableAmountDF.iloc[:, pivotTableAmountDF.columns.get_level_values(2) == "Total 3_/"]

pivotTableAmountNewColumnOrder = pd.concat(
    [pivotTableAmountCol1, pivotTableAmountCol2, pivotTableAmountCol3, pivotTableAmountCol4, pivotTableAmountCol5,
     pivotTableAmountCol6], axis=1)

# print(pivotTableAmountNewColumnOrder.head(10))

# For Cost.
pivotTableCostDF = pd.DataFrame(pivotTable.iloc[:, pivotTable.columns.get_level_values(1) == "Costo"])

# print(pivotTableCostDF.columns)
pivotTableCostDF[("Valor", "Costo", "Total 3_/")] = pivotTableCostDF[("Valor", "Costo", "Vista")] + \
                                                    pivotTableCostDF[("Valor", "Costo", "Plazo")] + \
                                                    pivotTableCostDF[("Valor", "Costo", "Bancario 1_/")] + \
                                                    pivotTableCostDF[("Valor", "Costo", "Interbancario 2_/")] + \
                                                    pivotTableCostDF[("Valor", "Costo", "Otros recursos")]

pivotTableCostCol1 = pivotTableCostDF.iloc[:, pivotTableCostDF.columns.get_level_values(2) == "Vista"]
pivotTableCostCol2 = pivotTableCostDF.iloc[:, pivotTableCostDF.columns.get_level_values(2) == "Plazo"]
pivotTableCostCol3 = pivotTableCostDF.iloc[:, pivotTableCostDF.columns.get_level_values(2) == "Bancario 1_/"]
pivotTableCostCol4 = pivotTableCostDF.iloc[:, pivotTableCostDF.columns.get_level_values(2) == "Interbancario 2_/"]
pivotTableCostCol5 = pivotTableCostDF.iloc[:, pivotTableCostDF.columns.get_level_values(2) == "Otros recursos"]
pivotTableCostCol6 = pivotTableCostDF.iloc[:, pivotTableCostDF.columns.get_level_values(2) == "Total 3_/"]

pivotTableCostNewColumnOrder = pd.concat(
    [pivotTableCostCol1, pivotTableCostCol2, pivotTableCostCol3, pivotTableCostCol4, pivotTableCostCol5,
     pivotTableCostCol6], axis=1)

# print(pivotTableCostNewColumnOrder.head(10))

# Joining amount and cost.
pivotTableAmountCost = pd.concat([pivotTableAmountNewColumnOrder, pivotTableCostNewColumnOrder], axis=1)

# print(pivotTableAmountCost.head(10))

# Reordering headers.
pivotTableAmountCostLevelReorder = pivotTableAmountCost.reorder_levels([2, 1, 0], axis=1)

# print(pivotTableAmountCostLevelReorder.head(10))
# print(pivotTableAmountCostLevelReorder.columns)

# Removing "Valor" header.
pivotTableAmountCostHeaderRemove = pivotTableAmountCostLevelReorder.droplevel(2, axis=1)

# print(pivotTableAmountCostHeaderRemove.head(100))
# print(pivotTableAmountCostHeaderRemove.columns)
# print(pivotTableAmountCostHeaderRemove.index)

# Compute totals.
vistaImporteTotal = pivotTableAmountCostHeaderRemove[("Vista", "Importe")].sum()
plazoImporteTotal = pivotTableAmountCostHeaderRemove[("Plazo", "Importe")].sum()
bancarioImporteTotal = pivotTableAmountCostHeaderRemove[("Bancario 1_/", "Importe")].sum()
interbancarioImporteTotal = pivotTableAmountCostHeaderRemove[("Interbancario 2_/", "Importe")].sum()
otrosRecursosImporteTotal = pivotTableAmountCostHeaderRemove[("Otros recursos", "Importe")].sum()
totalImporteTotal = pivotTableAmountCostHeaderRemove[("Total 3_/", "Importe")].sum()

vistaCostoTotal = pivotTableAmountCostHeaderRemove[("Vista", "Costo")].sum()
plazoCostoTotal = pivotTableAmountCostHeaderRemove[("Plazo", "Costo")].sum()
bancarioCostoTotal = pivotTableAmountCostHeaderRemove[("Bancario 1_/", "Costo")].sum()
interbancarioCostoTotal = pivotTableAmountCostHeaderRemove[("Interbancario 2_/", "Costo")].sum()
otrosRecursosCostoTotal = pivotTableAmountCostHeaderRemove[("Otros recursos", "Costo")].sum()
totalCostoTotal = pivotTableAmountCostHeaderRemove[("Total 3_/", "Costo")].sum()

# Add grand total row.
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Vista", "Importe")] = vistaImporteTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Plazo", "Importe")] = plazoImporteTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Bancario 1_/", "Importe")] = bancarioImporteTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Interbancario 2_/", "Importe")] = interbancarioImporteTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Otros recursos", "Importe")] = otrosRecursosImporteTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Total 3_/", "Importe")] = totalImporteTotal

pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Vista", "Costo")] = vistaCostoTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Plazo", "Costo")] = plazoCostoTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Bancario 1_/", "Costo")] = bancarioCostoTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Interbancario 2_/", "Costo")] = interbancarioCostoTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Otros recursos", "Costo")] = otrosRecursosCostoTotal
pivotTableAmountCostHeaderRemove.at[
    ("2021-02-28", "TOT. SISTEMA BANCA MÚLTIPLE"), ("Total 3_/", "Costo")] = totalCostoTotal

# Prints the result table.
print(pivotTableAmountCostHeaderRemove)

# Writes to HTML the pivot table.
pivotTableAmountCostHeaderRemove.to_html(outputPath)
