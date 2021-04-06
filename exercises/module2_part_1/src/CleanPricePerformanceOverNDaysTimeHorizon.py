import os
import pandas as pd

"""
Calculate the performance of the Clean Price over a two-day time horizon using "historico-vector_m.csv" data.
"""


def coupon_performance_over_n_days_time_horizon(dataFrameRow, nDaysTimeHorizon=2):
    couponPerformanceOverNDaysTimeHorizon = 0.0
    if dataFrameRow["DIAS X VENCER"] >= nDaysTimeHorizon:
        couponPerformanceOverNDaysTimeHorizon = (100 * (nDaysTimeHorizon * (dataFrameRow["TASA CUPON"] / 360)))
        return couponPerformanceOverNDaysTimeHorizon
    else:
        return "Error: The expiration days are less than the time horizon days."


def clean_price_performance(dataFrameRow):
    cleanPricePerformance = 0.0
    if dataFrameRow["RENDIMIENTO CUPON A N DIAS"] != "Error: The expiration days are less than the time horizon days.":
        cleanPricePerformance = ((dataFrameRow["PRECIO LIMPIO"] + dataFrameRow["RENDIMIENTO CUPON A N DIAS"]) /
                                 dataFrameRow["PRECIO LIMPIO"]) - 1
    else:
        return "Error: Can not compute value."
    return cleanPricePerformance


# Sets Pandas options.
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.6f" % x)

# inputPath to the CSV file.
inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science", "exercises",
                         "module2_part_1", "data", "historico-vector_m.csv")

# outputPath to the CSV result file.
outputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science", "exercises",
                          "module2_part_1", "output", "rendimiento-precio-limpio.csv")

# Prints the absolute inputPath to the CSV file.
print("The input CSV file is: ", inputPath)

# Reads the CSV data file.
historicalMVectorCSV = pd.read_csv(inputPath, dtype="str", encoding="ISO-8859-1")

# Converts to dataframe.
historicalMVectorDF = pd.DataFrame(historicalMVectorCSV)

# Dataframe size.
print("Original dataframe size:", historicalMVectorDF.shape)

# Sets the type of each column of the data frame.
historicalMVectorDF["INSTRUMENTO"] = historicalMVectorDF["INSTRUMENTO"].astype(str)
historicalMVectorDF["FECHA"] = pd.to_datetime(historicalMVectorDF["FECHA"], dayfirst=True)
historicalMVectorDF["TIPO VALOR"] = historicalMVectorDF["TIPO VALOR"].astype(str)
historicalMVectorDF["PRECIO SUCIO"] = historicalMVectorDF["PRECIO SUCIO"].astype(float)
historicalMVectorDF["PRECIO LIMPIO"] = historicalMVectorDF["PRECIO LIMPIO"].astype(float)
historicalMVectorDF["DIAS X VENCER"] = historicalMVectorDF["DIAS X VENCER"].astype(int)
historicalMVectorDF["RENDIMIENTO"] = historicalMVectorDF["RENDIMIENTO"].astype(float)
historicalMVectorDF["TASA CUPON"] = historicalMVectorDF["TASA CUPON"].astype(float)

# Assigns the data frame indexes.
historicalMVectorDF.set_index(["INSTRUMENTO", "FECHA"], inplace=True)

# Sorts the data frame by its indexes.
historicalMVectorDF.sort_values(["INSTRUMENTO", "FECHA"], ascending=True, inplace=True)

# Asks for number of days time horizon to compute for.
numberDaysTimeHorizon = int(input("Number of days time horizon: "))

# Validate number of days time horizon.
if numberDaysTimeHorizon <= 0:
    numberDaysTimeHorizon = 2
elif numberDaysTimeHorizon > 182:
    numberDaysTimeHorizon = 2

# Compute the clean price performance.
historicalMVectorDF["RENDIMIENTO CUPON A N DIAS"] = historicalMVectorDF.apply(
    lambda dataFrameRow: coupon_performance_over_n_days_time_horizon(dataFrameRow, numberDaysTimeHorizon), axis=1)

historicalMVectorDF["RENDIMIENTO PRECIO LIMPIO"] = historicalMVectorDF.apply(
    lambda dataFrameRow: clean_price_performance(dataFrameRow), axis=1)

historicalMVectorDF = historicalMVectorDF.loc[(historicalMVectorDF[
                                                   "RENDIMIENTO CUPON A N DIAS"] != "Error: The expiration days are less than the time horizon days.") & (
                                                      historicalMVectorDF[
                                                          "RENDIMIENTO PRECIO LIMPIO"] != "Error: Can not compute value.")]
print("Result dataframe size:", historicalMVectorDF.shape)

# Output the result dataframe to a CSV file.
historicalMVectorDF.to_csv(outputPath)

print("Your result output csv file is: ", outputPath)
