import tkinter as tk

import joblib
import numpy as np
import pandas as pd

from DataScaler import scaleGUIData
from FileManager import getOutputPath
from Matrix import predict

# Load computed model.
regressor = joblib.load(getOutputPath("google-stock-price-trained-model.pkl"))

print("Intercept: \n", regressor.intercept_)
print("Coefficients: \n", regressor.coef_)

# tkinter GUI.
root = tk.Tk()

guiCanvas = tk.Canvas(root, width=500, height=300)
guiCanvas.pack()

# With sklearn.
Intercept_result = ("Intercept: ", regressor.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify="center")
guiCanvas.create_window(260, 220, window=label_Intercept)

# With sklearn.
Coefficients_result = ("Coefficients: ", regressor.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify="center")
guiCanvas.create_window(260, 240, window=label_Coefficients)

# High stock price label and input box.
highStockPriceLabel = tk.Label(root, text="Type high stock price: ")
guiCanvas.create_window(100, 100, window=highStockPriceLabel)

highStockPriceTextEntry = tk.Entry(root)  # create 1st entry box
guiCanvas.create_window(270, 100, window=highStockPriceTextEntry)

# Low stock price label and input box.
lowStockPriceLabel = tk.Label(root, text=" Type low stock price: ")
guiCanvas.create_window(100, 120, window=lowStockPriceLabel)

lowStockPriceTextEntry = tk.Entry(root)  # create 2nd entry box
guiCanvas.create_window(270, 120, window=lowStockPriceTextEntry)

# Open stock price label and input box.
openStockPriceLabel = tk.Label(root, text=" Type open stock price: ")
guiCanvas.create_window(100, 140, window=openStockPriceLabel)

openStockPriceTextEntry = tk.Entry(root)  # create 2nd entry box
guiCanvas.create_window(270, 140, window=openStockPriceTextEntry)

# Volume stock label and input box.
stockVolumeLabel = tk.Label(root, text=" Type stock volume: ")
guiCanvas.create_window(100, 160, window=stockVolumeLabel)

stockVolumeTextEntry = tk.Entry(root)  # create 2nd entry box
guiCanvas.create_window(270, 160, window=stockVolumeTextEntry)


def computePrediction():
    global highStockPrice  # High stock price.
    highStockPrice = float(highStockPriceTextEntry.get())

    global lowStockPrice  # Low stock price.
    lowStockPrice = float(lowStockPriceTextEntry.get())

    global openStockPrice  # Open stock price.
    openStockPrice = float(openStockPriceTextEntry.get())

    global stockVolume  # Stock volume.
    stockVolume = float(stockVolumeTextEntry.get())

    # Setup and scale values.
    if (highStockPrice == 0 and lowStockPrice == 0 and openStockPrice == 0 and stockVolume == 0):
        X = np.zeros(4)
    else:
        scaler = joblib.load(getOutputPath("google-stock-price-scaler.pkl"))
        X = pd.DataFrame([[highStockPrice, lowStockPrice, openStockPrice, stockVolume]])
        X.columns = ["high", "low", "open", "volume"]
        X = scaleGUIData(X, scaler)

    # Compute predicted value.
    y = predict(regressor.coef_, X, regressor.intercept_)
    print("y=f(X)=", y)

    predictionResult = ("Predicted close stock price (USD)", y)
    # predictionResult = ("Predicted close stock price (USD)", regressor.predict(X))
    predictionLabel = tk.Label(root, text=predictionResult, bg="blue", fg="white")
    guiCanvas.create_window(260, 280, window=predictionLabel)


predictButton = tk.Button(root, text="Predict close stock price", command=computePrediction,
                          bg="gray", fg="black")
guiCanvas.create_window(270, 200, window=predictButton)

root.mainloop()
