#!/usr/bin/python
#coding: utf8

import sys
import numpy as np
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

#Inicializa las ventanas
qtCreatorFile = "C:\\Users\\ricardo\\Desktop\\PantallasPython\\PantallasQT\\Graficas.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class PantallaGraficas(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self):
		super().__init__()
		QtWidgets.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		self.ui = Ui_MainWindow() #variable correspondiente a la pantalla principal
		self.ui.setupUi(self) #importante para reconocer los propios objetos de la pantalla
		#Eliminar barra
		#self.setWindowFlag(Qt.FramelessWindowHint)
		#Transparente
		#self.setAttribute(Qt.WA_TranslucentBackground)
		self.setWindowTitle("Gráficas")
		#Definición de gráfica
		self.grafica1 = Canvas_grafica1()
		self.grafica2 = Canvas_grafica2()
		#Posicionamiento en objeto de pantalla
		self.ui.Grafica1.addWidget(self.grafica1)
		self.ui.Grafica2.addWidget(self.grafica2)
		self.show()
class Canvas_grafica1(FigureCanvas):
	def __init__(self, parent=None):
		self.fig, self.ax = plt.subplots(1,dpi=100,figsize=(5,5),sharey=True,facecolor="white")
		super().__init__(self.fig)
		nombres = ["15","25","30","35","40"]
		colores = ["red","red","red","red","red"]
		tamano = [10,15,20,25,30]
		self.ax.bar(nombres,tamano,color=colores)
		self.fig.suptitle("Gráfica de Barras",size=9)

class Canvas_grafica2(FigureCanvas):
	def __init__(self, parent=None):
		self.fig, self.ax = plt.subplots(1,dpi=100,figsize=(5,5),sharey=True,facecolor="white")
		super().__init__(self.fig)
		x = [1,2,3,4,5,6,7]
		y1 = [1,0,1,3,2,4,3]
		y2 = [0,2,2,3,4,5,6]
		y3 = [3,1,3,4,2,7,6]
		y = np.vstack([y1,y2,y3])
		labels = ["Y1","Y2","Y3"]
		color=["orange","blue","green"]
		self.ax.stackplot(x,y)
		self.fig.suptitle("Grafica Stackplot",size=9)

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = PantallaGraficas()
	app.exec_()
