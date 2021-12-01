from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from FileManager import getOutputPath


# Distribution graphs (histogram/bar graph) of column data.
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    # df = df[[col for col in df if nunique[col] > 1 and nunique[
    #     col] < 50]]  # For displaying purposes, pick columns that have between 1 and 50 unique values.
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = round((nCol + nGraphPerRow - 1) / nGraphPerRow)
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor="w", edgecolor="k")
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel("counts")
        plt.xticks(rotation=90)
        plt.title(f"{columnNames[i]} (column {i})")
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.savefig(getOutputPath("histogram-per-column.png"))
    plt.show()


# Correlation matrix.
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna("columns")  # drop columns with NaN.
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values.
    if df.shape[1] < 2:
        print(f"No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2")
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor="w", edgecolor="k")
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f"Correlation Matrix for {filename}", fontsize=15)
    plt.savefig(getOutputPath("correlation-matrix.png"))
    plt.show()


# Scatter and density plots.
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])  # keep only numerical columns.
    # Remove rows and columns that would lead to df being singular.
    df = df.dropna("columns")
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values.
    columnNames = list(df)
    if len(columnNames) > 10:  # reduce the number of columns for matrix inversion of kernel density plots.
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal="kde")
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate("Corr. coef = %.3f" % corrs[i, j], (0.8, 0.2), xycoords="axes fraction", ha="center",
                          va="center", size=textSize)
    plt.suptitle("Scatter and Density Plot")
    plt.savefig(getOutputPath("scatter-and-density-plots.png"))
    plt.show()


# Heat map.
def heatMapPlot(df, corrVar):
    sns.set(palette="YlGnBu", font="Serif", style="white",
            rc={"axes.facecolor": "whitesmoke", "figure.facecolor": "whitesmoke"})

    df.corr()[corrVar]

    fig = plt.figure(figsize=(15, 8))
    heatMap = sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", linecolor='white', linewidth=2)
    figure = heatMap.get_figure()
    figure.savefig(getOutputPath("heat-map.png"), dpi=400)


# Plot by month.
def plotByMonth(df):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df.index,
        y=df["open"],
        name="Stock Open Price",
        marker_color="green"))
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["close"],
        name="Stock Close Price",
        marker_color="blue"))
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["adjOpen"],
        name="Stock adjOpen Price",
        marker_color="lightgreen"))
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["adjClose"],
        name="Stock adjClose Price",
        marker_color="lightblue"))

    fig.update_layout(barmode="group", xaxis_tickangle=-45,
                      title="Google stock price comparison between Stock open and close price grouped by month")
    fig.show()


# Plot low and high stock price.
def plotLowAndHighStock(df, monthIndex):
    # Grouping by high and low price.
    df.groupby(df["date"].dt.strftime("%B"))[
        "low"].min()

    df_high = \
        df.groupby(df["date"].dt.strftime("%B"))[
            "high"].max()
    df_high = df_high.reindex(monthIndex, axis=0)

    df_adjHigh = \
        df.groupby(df["date"].dt.strftime("%B"))[
            "adjHigh"].max()
    df_adjHigh = df_adjHigh.reindex(monthIndex, axis=0)

    df_low = \
        df.groupby(df["date"].dt.strftime("%B"))[
            "low"].min()
    df_low = df_low.reindex(monthIndex, axis=0)

    df_adjLow = \
        df.groupby(df["date"].dt.strftime("%B"))[
            "adjLow"].min()
    df_adjLow = df_adjLow.reindex(monthIndex, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_high.index,
        y=df_high,
        name="Stock high Price",
        marker_color="orange"
    ))
    fig.add_trace(go.Bar(
        x=df_low.index,
        y=df_low,
        name="Stock low Price",
        marker_color="yellow"
    ))
    fig.add_trace(go.Bar(
        x=df_adjHigh.index,
        y=df_adjHigh,
        name="Stock adjHigh Price",
        marker_color="orangered"
    ))
    fig.add_trace(go.Bar(
        x=df_adjLow.index,
        y=df_adjLow,
        name="Stock adjLow Price",
        marker_color="yellowgreen"
    ))

    fig.update_layout(barmode="group",
                      title="Google stock price comparison between Stock high and low price grouped by month")
    fig.show()


# Plot stock price comparison chart.
def plotStockPriceComparisonChart(df):
    legendNames = cycle(
        ["Stock Open Price", "Stock Close Price", "Stock High Price", "Stock Low Price", "Stock adjOpen Price",
         "Stock adjClose Price", "Stock adjHigh Price", "Stock adjLow Price"])

    fig = px.line(df, x=df.date,
                  y=[df["open"], df["close"],
                     df["high"], df["low"],
                     df["adjOpen"], df["adjClose"],
                     df["adjHigh"], df["adjLow"]],
                  labels={"date": "Date", "value": "Stock value"})
    fig.update_layout(title_text="Stock analysis chart", font_size=15, font_color="black",
                      legend_title_text="Stock Parameters")
    fig.for_each_trace(lambda t: t.update(name=next(legendNames)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.show()


# Plot only stock close price.
def plotStockClosePrice(df):
    closePriceDataFrame = df[["date", "close"]]
    print("Shape of stock close price dataframe:", closePriceDataFrame.shape)

    # Plot stock close price.
    fig = px.line(closePriceDataFrame, x=closePriceDataFrame.date,
                  y=closePriceDataFrame.close, labels={"date": "Date", "close": "Stock close price"})
    fig.update_traces(marker_line_width=2, opacity=0.8)
    fig.update_layout(title_text="Stock close price chart", plot_bgcolor="white", font_size=15, font_color="black")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.show()
