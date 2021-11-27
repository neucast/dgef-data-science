# Exploratory Data Analysis.

import warnings
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from sklearn import set_config
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from DataPlotter import heatMapPlot
from FileManager import getInputPath

# Configure.
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Data set path.
inputPath = getInputPath("base-final-regresion.csv")

# Reads the CSV data file.
googleStockPriceDataSet = pd.read_csv(inputPath, delimiter=",", encoding="ISO-8859-1")
googleStockPriceDataSet.dataframeName = "base-final-regresion.csv"
nRow, nCol = googleStockPriceDataSet.shape
print(f"There are {nRow} rows and {nCol} columns in the data set.")

# Exploring the data set.
print(googleStockPriceDataSet.head(5))
googleStockPriceDataSet.info()  # I can observe that there are no missing values and NOT all are numerical (symbol, date).
print(googleStockPriceDataSet.columns)

print("Total number of days: ", googleStockPriceDataSet.shape[0])
print("Total number of variables: ", googleStockPriceDataSet.shape[1])

# Descriptive statistic.
print(googleStockPriceDataSet.describe())

# Verifying null and NA data.
print("Searching for null values...\n")
nullValuesFound = googleStockPriceDataSet.isnull().sum()
print(nullValuesFound)  # No null values found.

print("Searching for NA values...\n")
naValuesFound = googleStockPriceDataSet.isna().values.any()
print("NA values:", naValuesFound)  # No NA values found.

# Getting the dataframe.
googleStockPriceDataFrame = pd.DataFrame(googleStockPriceDataSet)

# Convert date field from string to Date format.
googleStockPriceDataFrame["date"] = pd.to_datetime(googleStockPriceDataFrame.date, infer_datetime_format=True)
print(googleStockPriceDataFrame.head())

# Sort by date.
googleStockPriceOrderedDataFrame = googleStockPriceDataFrame.sort_values(by="date")

# EDA.
print("Starting date: ", googleStockPriceOrderedDataFrame["date"].min())
print("Ending date: ", googleStockPriceOrderedDataFrame["date"].max())
print("Duration: ",
      googleStockPriceOrderedDataFrame["date"].max() - googleStockPriceOrderedDataFrame["date"].min())

# Distribution graphs (histogram/bar graph) of sampled columns.
# plotPerColumnDistribution(googleStockPriceDataSet, 14, 7)

# Correlation matrix.
# plotCorrelationMatrix(googleStockPriceDataSet, 15)

# Scatter and density plots.
# plotScatterMatrix(googleStockPriceDataSet, 15, 5)

# Grouping the data by month.
googleStockPriceGroupedByMonth = \
    googleStockPriceOrderedDataFrame.groupby(googleStockPriceOrderedDataFrame["date"].dt.strftime("%B"))[
        ["open", "close", "adjOpen", "adjClose"]].mean()
monthIndex = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
googleStockPriceGroupedByMonth = googleStockPriceGroupedByMonth.reindex(monthIndex, axis=0)
print(googleStockPriceGroupedByMonth)

# Plot stock open and close price grouped by month
fig = go.Figure()

fig.add_trace(go.Bar(
    x=googleStockPriceGroupedByMonth.index,
    y=googleStockPriceGroupedByMonth["open"],
    name="Stock Open Price",
    marker_color="green"))
fig.add_trace(go.Bar(
    x=googleStockPriceGroupedByMonth.index,
    y=googleStockPriceGroupedByMonth["close"],
    name="Stock Close Price",
    marker_color="blue"))
fig.add_trace(go.Bar(
    x=googleStockPriceGroupedByMonth.index,
    y=googleStockPriceGroupedByMonth["adjOpen"],
    name="Stock adjOpen Price",
    marker_color="lightgreen"))
fig.add_trace(go.Bar(
    x=googleStockPriceGroupedByMonth.index,
    y=googleStockPriceGroupedByMonth["adjClose"],
    name="Stock adjClose Price",
    marker_color="lightblue"))

fig.update_layout(barmode="group", xaxis_tickangle=-45,
                  title="Google stock price comparison between Stock open and close price grouped by month")
fig.show()

# Grouping by high and low price.
googleStockPriceOrderedDataFrame.groupby(googleStockPriceOrderedDataFrame["date"].dt.strftime("%B"))[
    "low"].min()

googleStockPriceGroupedByMonth_high = \
    googleStockPriceOrderedDataFrame.groupby(googleStockPriceOrderedDataFrame["date"].dt.strftime("%B"))[
        "high"].max()
googleStockPriceGroupedByMonth_high = googleStockPriceGroupedByMonth_high.reindex(monthIndex, axis=0)

googleStockPriceGroupedByMonth_adjHigh = \
    googleStockPriceOrderedDataFrame.groupby(googleStockPriceOrderedDataFrame["date"].dt.strftime("%B"))[
        "adjHigh"].max()
googleStockPriceGroupedByMonth_adjHigh = googleStockPriceGroupedByMonth_adjHigh.reindex(monthIndex, axis=0)

googleStockPriceGroupedByMonth_low = \
    googleStockPriceOrderedDataFrame.groupby(googleStockPriceOrderedDataFrame["date"].dt.strftime("%B"))[
        "low"].min()
googleStockPriceGroupedByMonth_low = googleStockPriceGroupedByMonth_low.reindex(monthIndex, axis=0)

googleStockPriceGroupedByMonth_adjLow = \
    googleStockPriceOrderedDataFrame.groupby(googleStockPriceOrderedDataFrame["date"].dt.strftime("%B"))[
        "adjLow"].min()
googleStockPriceGroupedByMonth_adjLow = googleStockPriceGroupedByMonth_adjLow.reindex(monthIndex, axis=0)

# Plot stock low and high stock price grouped by month.
fig = go.Figure()
fig.add_trace(go.Bar(
    x=googleStockPriceGroupedByMonth_high.index,
    y=googleStockPriceGroupedByMonth_high,
    name="Stock high Price",
    marker_color="orange"
))
fig.add_trace(go.Bar(
    x=googleStockPriceGroupedByMonth_low.index,
    y=googleStockPriceGroupedByMonth_low,
    name="Stock low Price",
    marker_color="yellow"
))
fig.add_trace(go.Bar(
    x=googleStockPriceGroupedByMonth_adjHigh.index,
    y=googleStockPriceGroupedByMonth_adjHigh,
    name="Stock adjHigh Price",
    marker_color="orangered"
))
fig.add_trace(go.Bar(
    x=googleStockPriceGroupedByMonth_adjLow.index,
    y=googleStockPriceGroupedByMonth_adjLow,
    name="Stock adjLow Price",
    marker_color="yellowgreen"
))

fig.update_layout(barmode="group",
                  title="Google stock price comparison between Stock high and low price grouped by month")
fig.show()

# Stock price comparison chart.
legendNames = cycle(
    ["Stock Open Price", "Stock Close Price", "Stock High Price", "Stock Low Price", "Stock adjOpen Price",
     "Stock adjClose Price", "Stock adjHigh Price", "Stock adjLow Price"])

fig = px.line(googleStockPriceOrderedDataFrame, x=googleStockPriceOrderedDataFrame.date,
              y=[googleStockPriceOrderedDataFrame["open"], googleStockPriceOrderedDataFrame["close"],
                 googleStockPriceOrderedDataFrame["high"], googleStockPriceOrderedDataFrame["low"],
                 googleStockPriceOrderedDataFrame["adjOpen"], googleStockPriceOrderedDataFrame["adjClose"],
                 googleStockPriceOrderedDataFrame["adjHigh"], googleStockPriceOrderedDataFrame["adjLow"]],
              labels={"date": "Date", "value": "Stock value"})
fig.update_layout(title_text="Stock analysis chart", font_size=15, font_color="black",
                  legend_title_text="Stock Parameters")
fig.for_each_trace(lambda t: t.update(name=next(legendNames)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()

# Stock close price dataframe.
googleStockClosePriceDataFrame = googleStockPriceOrderedDataFrame[["date", "close"]]
print("Shape of stock close price dataframe:", googleStockClosePriceDataFrame.shape)

# Plot stock close price.
fig = px.line(googleStockClosePriceDataFrame, x=googleStockClosePriceDataFrame.date,
              y=googleStockClosePriceDataFrame.close, labels={"date": "Date", "close": "Stock close price"})
fig.update_traces(marker_line_width=2, opacity=0.8)
fig.update_layout(title_text="Stock close price chart", plot_bgcolor="white", font_size=15, font_color="black")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Stock volume dataframe.
googleStockVolumeDataFrame = go.Scatter(x=googleStockPriceOrderedDataFrame["date"],
                                        y=googleStockPriceOrderedDataFrame["volume"].values, name="Volume")

# Plot stock volume.
layout = go.Layout(dict(title="Google stock volume",
                        xaxis=dict(title="Year"),
                        yaxis=dict(title="Volume"),
                        ), legend=dict(
    orientation="h"))
py.iplot(dict(data=googleStockVolumeDataFrame, layout=layout), filename="basic-line")

# Stock price diff standard deviation.
googleStockPriceDataSet["price_diff"] = googleStockPriceDataSet["close"] - googleStockPriceDataSet["open"]
print(googleStockPriceDataSet["price_diff"].describe())
groupedByDate = googleStockPriceDataSet.groupby("date").agg({"price_diff": ["std", "min"]}).reset_index()
print(f"Average standard deviation of price change within a day in {groupedByDate['price_diff']['std'].mean():.4f}.")

# Plot close stock price correlation heat map.
googleStockPriceDataFrame.drop(["symbol"], axis=1, inplace=True)
heatMapPlot(googleStockPriceDataFrame, "close")

# Data sample.

# -----------Test 1---------------------
# Regressor score =  100.0
# Regressor r2 score =  1.0
# Regressor mean square error =  98.24
# Regressor mean absolute error =  6.95
# modelVariables = ["date", "volume", "open", "close", "high", "low"]
# independentVariables = ["date", "volume", "open", "high", "low"]
# --------------------------------

# -----------Test 2---------------------
# Regressor score =  100.0
# Regressor r2 score =  1.0
# Regressor mean square error =  96.58
# Regressor mean absolute error =  6.9
# modelVariables = ["date", "open", "close", "high", "low"]
# independentVariables = ["date", "open", "high", "low"]
# --------------------------------

# -----------Test 3: Excellent!---------------------
# Regressor score =  100.0
# Regressor r2 score =  1.0
# Regressor mean square error =  4.35
# Regressor mean absolute error =  1.45
modelVariables = ["date", "close", "adjVolume", "adjOpen", "adjClose", "adjHigh", "adjLow"]
independentVariables = ["date", "adjVolume", "adjOpen", "adjClose", "adjHigh", "adjLow"]
# --------------------------------

# -----------Test 4---------------------
# Regressor score =  100.0
# Regressor r2 score =  1.0
# Regressor mean square error =  6.31
# Regressor mean absolute error =  1.78
# modelVariables = ["date", "close", "adjOpen", "adjClose", "adjHigh", "adjLow"]
# independentVariables = ["date", "adjOpen", "adjClose", "adjHigh", "adjLow"]
# --------------------------------

# -----------Test 5---------------------
# Regressor score =  100.0
# Regressor r2 score =  1.0
# Regressor mean square error =  11.41
# Regressor mean absolute error =  2.35
# modelVariables = ["date", "volume", "open", "close", "high", "low", "adjVolume", "adjOpen", "adjClose", "adjHigh",
#                   "adjLow"]
# independentVariables = ["date", "volume", "open", "high", "low", "adjVolume", "adjOpen", "adjClose", "adjHigh",
#                         "adjLow"]
# --------------------------------

# -----------Test 6: Good!---------------------
# Regressor score =  100.0
# Regressor r2 score =  1.0
# Regressor mean square error =  11.32
# Regressor mean absolute error =  2.34
# modelVariables = ["date", "open", "close", "high", "low", "adjOpen", "adjClose", "adjHigh", "adjLow"]
# independentVariables = ["date", "open", "high", "low", "adjOpen", "adjClose", "adjHigh", "adjLow"]
# --------------------------------

modelDataFrame = googleStockPriceOrderedDataFrame[modelVariables]
modelDataFrame["date"] = pd.to_numeric(modelDataFrame.date)

test_sample_size = 0.2
X_split_training_set, X_split_testing_set, Y_split_training_set, Y_split_testing_set = train_test_split(
    modelDataFrame[independentVariables],
    modelDataFrame[["close"]],
    test_size=test_sample_size,
    random_state=1)

# Transforming data as a Numpy array.
X_training_set = np.asarray(
    X_split_training_set[independentVariables].copy(deep=True).reset_index(drop=True))
X_testing_set = np.asarray(
    X_split_testing_set[independentVariables].copy(deep=True).reset_index(drop=True))
Y_training_set = np.asarray(Y_split_training_set[["close"]].copy(deep=True).reset_index(drop=True))
Y_testing_set = np.asarray(Y_split_testing_set[["close"]].copy(deep=True).reset_index(drop=True))

# ---------------------------------- Linear Regression ----------------------------------
print("----------------------------- Linear Regression -----------------------------")
regressor = LinearRegression()
regressor.fit(X_training_set, Y_training_set)

print("Model intercept:", regressor.intercept_)
print("Model coefficients:", regressor.coef_)

set_config(display="diagram")

Y_predicted_set = regressor.predict(X_testing_set)

# Regressor score.
regressorScore = np.round(regressor.score(X_testing_set, Y_testing_set), 2) * 100
print("Regressor score = ", regressorScore)

# r2_score.
regressorR2Score = np.round(r2_score(Y_testing_set, Y_predicted_set), 2)
print("Regressor r2 score = ", regressorR2Score)

# Regressor MSE.
regressorMeanSquareError = np.round(mean_squared_error(Y_testing_set, Y_predicted_set), 2)
print("Regressor mean square error = ", regressorMeanSquareError)

# Regressor MAE.
regressorMeanAbsoluteError = np.round(mean_absolute_error(Y_testing_set, Y_predicted_set), 2)
print("Regressor mean absolute error = ", regressorMeanAbsoluteError)

# Predicted vs Actual stock closing prices plot.
xTestingDataFrame = pd.DataFrame(X_testing_set)
xTestingDataFrame = pd.DataFrame(X_testing_set, xTestingDataFrame.index, independentVariables)
xTestingDataFrame["date"] = pd.to_datetime(xTestingDataFrame.date, infer_datetime_format=True)

yTestingDataFrame = pd.DataFrame(Y_testing_set)

predictedDataFrame = pd.DataFrame(Y_predicted_set, yTestingDataFrame.index, ["prediction"])
yTestingDataFrame = pd.DataFrame(Y_testing_set, yTestingDataFrame.index, ["close"])

xTestingDataFrame = xTestingDataFrame.join(predictedDataFrame)
xTestingDataFrame = xTestingDataFrame.join(yTestingDataFrame)
xTestingDataFrame["diff"] = xTestingDataFrame["close"] - xTestingDataFrame["prediction"]
xTestingDataFrame = xTestingDataFrame.sort_values(by="date")
print(xTestingDataFrame.head())

plotData = []
plotData.append(go.Scatter(x=xTestingDataFrame["date"], y=xTestingDataFrame["prediction"].values, name="Prediction"))
plotData.append(go.Scatter(x=xTestingDataFrame["date"], y=xTestingDataFrame["close"].values, name="Actual"))
layout = go.Layout(dict(title="Predicted and Actual stock closing prices of Google",
                        xaxis=dict(title="Year"),
                        yaxis=dict(title="Price (USD)"),
                        ), legend=dict(orientation="h"))
py.iplot(dict(data=plotData, layout=layout), filename="basic-line")
# ------------------------------------------------------------------------------------------------------
