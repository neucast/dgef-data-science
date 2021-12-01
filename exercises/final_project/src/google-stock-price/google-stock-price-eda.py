# Exploratory Data Analysis.

import warnings

import pandas as pd

from DataPlotter import heatMapPlot, plotPerColumnDistribution, plotCorrelationMatrix, plotScatterMatrix, plotByMonth, \
    plotLowAndHighStock, plotStockPriceComparisonChart, plotStockClosePrice
from FileManager import getInputPath

# Configure.
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Data set path.
inputPath = getInputPath("base-final-regresion.csv")

# Reads the CSV data file.
googleStockPriceDataFrame = pd.read_csv(inputPath, delimiter=",", encoding="ISO-8859-1")
googleStockPriceDataFrame.dataframeName = "base-final-regresion.csv"
nRow, nCol = googleStockPriceDataFrame.shape
print(f"There are {nRow} rows and {nCol} columns in the data set.")

# Exploring the data set.
print(googleStockPriceDataFrame.head(5))
googleStockPriceDataFrame.info()  # I can observe that there are no missing values and NOT all are numerical (symbol, date).
print(googleStockPriceDataFrame.columns)

print("Total number of days: ", googleStockPriceDataFrame.shape[0])
print("Total number of variables: ", googleStockPriceDataFrame.shape[1])

# Descriptive statistic.
print(googleStockPriceDataFrame.describe())

# Verifying null and NA data.
print("Searching for null values...\n")
nullValuesFound = googleStockPriceDataFrame.isnull().sum()
print(nullValuesFound)  # No null values found.

print("Searching for NA values...\n")
naValuesFound = googleStockPriceDataFrame.isna().values.any()
print("NA values:", naValuesFound)  # No NA values found.

# Distribution graphs (histogram/bar graph) of sampled columns.
plotPerColumnDistribution(googleStockPriceDataFrame, 14, 7)

# Correlation matrix.
plotCorrelationMatrix(googleStockPriceDataFrame, 15)

# Scatter and density plots.
plotScatterMatrix(googleStockPriceDataFrame, 15, 5)

# Convert date field from string to Date format.
googleStockPriceDataFrame["date"] = pd.to_datetime(googleStockPriceDataFrame.date, infer_datetime_format=True)
print(googleStockPriceDataFrame.head(5))

# Sort by date.
dataFrameOrderedByDate = googleStockPriceDataFrame.sort_values(by="date")

# EDA.
print("Starting date: ", dataFrameOrderedByDate["date"].min())
print("Ending date: ", dataFrameOrderedByDate["date"].max())
print("Duration: ", dataFrameOrderedByDate["date"].max() - dataFrameOrderedByDate["date"].min())

# Grouping the data by month.
dataFrameGroupedByMonth = dataFrameOrderedByDate.groupby(dataFrameOrderedByDate["date"].dt.strftime("%B"))[
    ["open", "close", "adjOpen", "adjClose"]].mean()

monthIndex = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]

dataFrameGroupedByMonth = dataFrameGroupedByMonth.reindex(monthIndex, axis=0)
print(dataFrameGroupedByMonth)

# Plot stock open and close price grouped by month.
plotByMonth(dataFrameGroupedByMonth)

# Plot low and high stock price grouped by month.
plotLowAndHighStock(dataFrameOrderedByDate, monthIndex)

# Plot stock price comparison chart.
plotStockPriceComparisonChart(dataFrameOrderedByDate)

# Plot stock close price dataframe.
plotStockClosePrice(dataFrameOrderedByDate)

# Stock price diff standard deviation.
googleStockPriceDataFrame["price_diff"] = googleStockPriceDataFrame["close"] - googleStockPriceDataFrame["open"]
print(googleStockPriceDataFrame["price_diff"].describe())
groupedByDate = googleStockPriceDataFrame.groupby("date").agg({"price_diff": ["std", "min"]}).reset_index()
print(f"Average standard deviation of price change within a day in {groupedByDate['price_diff']['std'].mean():.4f}.")

# Plot close stock price correlation heat map.
googleStockPriceDataFrame.drop(["symbol"], axis=1, inplace=True)
heatMapPlot(googleStockPriceDataFrame, "close")
