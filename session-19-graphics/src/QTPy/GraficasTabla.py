#!/usr/bin/python
#coding: utf8

import sys
import numpy as np
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.cm #Librería para establecer, entre otras opciones, varios colores en las gráficas
import pandas as pd

#Inicializa las ventanas
qtCreatorFile = "C:\\Users\\ricardo\\Desktop\\PantallasPython\\PantallasQT\\GraficaTabla.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class GraficaTablas(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self):
		super().__init__()
		QtWidgets.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		self.ui = Ui_MainWindow() #variable correspondiente a la pantalla principal
		self.setupUi(self) #importante para reconocer los propios objetos de la pantalla
		self.setFixedSize(self.size()) #Evitar que se pueda modificar el tamaño de la pantalla
		#Eliminar barra
		#self.setWindowFlag(Qt.FramelessWindowHint)
		#Transparente
		#self.setAttribute(Qt.WA_TranslucentBackground)
		self.setWindowTitle("Gráficas")
		#Definición de gráfica
		self.grafica = Canvas_grafica1()
		#Posicionamiento en objeto de pantalla
		self.Grafica.addWidget(self.grafica)
		#Llenado de tabla
		T = pd.read_csv("C:\\Users\\ricardo\\Desktop\\PantallasPython\\PantallasQT\\EjemploTablaDatos.csv",dtype='str',encoding = "ISO-8859-1")
		T.fillna(0,inplace=True)
		n_ren = len(T.index)
		n_col = len(T.columns)
        #Conteo del número de renglones y columnas y agregado en caso necesario
		#https://doc.qt.io/archives/qtforpython-5.14/PySide2/QtWidgets/QTableWidget.html#PySide2.QtWidgets.PySide2.QtWidgets.QTableWidget
		n_ren_def = self.Tabla.rowCount()
		n_col_def = self.Tabla.columnCount()
		if n_ren_def < n_ren:
			for i in range(n_ren - n_ren_def):
				self.Tabla.insertRow(n_ren_def + i)
		if n_col_def < n_col:
			for i in range(n_col - n_col_def):
				self.Tabla.insertColumn(n_col_def + i)
		for i in range(n_ren):
			for j in range(n_col):
				try:
					celda = QTableWidgetItem(str(round(float(T.iloc[i,j]),5)))
				except:
					celda = QTableWidgetItem(str(T.iloc[i,j]))
				celda.setTextAlignment(Qt.AlignCenter)
				self.Tabla.setItem(i,j,celda)
		#Nombre de las columnas en la Tabla
		self.Tabla.setHorizontalHeaderLabels(T.columns)
		self.show()
class Canvas_grafica1(FigureCanvas):
	def __init__(self, parent=None):
		self.fig, self.ax = plt.subplots(1,dpi=100,figsize=(15,15),sharey=True,facecolor="white")
		super().__init__(self.fig)
		T = pd.read_csv("C:\\Users\\ricardo\\Desktop\\PantallasPython\\PantallasQT\\EjemploTablaDatos.csv",dtype='str',encoding = "ISO-8859-1")
		vcol = T.columns.tolist()
		vcol.remove("Institucion")
		colores = ["darkred","blue","darkgreen","darkblue"]
		for i in range(len(vcol)):
                        columna = T[vcol[i]].astype(float).values.tolist()
                        numero_de_grupos = len(columna)
                        indice_barras = np.arange(numero_de_grupos)
                        ancho_barras =0.07
                        plt.bar(indice_barras + i*ancho_barras, columna, width=ancho_barras,color=colores[i%len(colores)])
		# Se colocan los indicadores en el eje x
		plt.xticks(indice_barras + 5*ancho_barras, vcol)
		#self.ax.bar(nombres,tamano,color=colores)
		self.fig.suptitle("Aforos",size=12)

if __name__ == "__main__":
        app = QtWidgets.QApplication(sys.argv)
        window = GraficaTablas()
        app.exec_()

#self.statTable.setSizeAdjustPolicy(
#        QtWidgets.QAbstractScrollArea.AdjustToContents)
