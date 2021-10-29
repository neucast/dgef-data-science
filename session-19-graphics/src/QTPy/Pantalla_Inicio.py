#!/usr/bin/python
#coding: utf8

import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt
#https://www.qt.io/

#Pantallas
import GraficasTabla

#Inicializa las ventanas
qtCreatorFile = "C:\\Users\\ricardo\\Desktop\\PantallasPython\\PantallasQT\\Credenciales.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self):
		super().__init__()
		QtWidgets.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		self.ui = Ui_MainWindow() #variable correspondiente a la pantalla principal
		#self.ui.setupUi(self)
		self.setupUi(self)
		#Eliminar barra
		self.setWindowFlag(Qt.FramelessWindowHint)
		#Transparente
		self.setAttribute(Qt.WA_TranslucentBackground)
		self.setWindowTitle("Pantalla crendenciales")
		self.BotonAceptar.clicked.connect(self.funcion1)
		self.BotonCerrar.clicked.connect(self.funcion2)
		self.Contrasena.setEchoMode(2)
		self.show()
	def funcion1(self):
		x = self.Usuario.text()
		y = self.Contrasena.text()
		if x == "K11135" and y == "123456":
			self.EtiqContra.setText(" ")
			self.EtiqUsuario.setText(" ")
			self.close()
			self.GraficasTablas = GraficasTabla.GraficaTablas()
		elif x == "K11135" and y != "123456":
			self.EtiqContra.setText("Contraseña incorrecta")
		elif x != "K11135" and y == "123456":
			self.EtiqUsuario.setText("Usuario incorrecto")
		else:
			self.EtiqUsuario.setText("Usuario incorrecto")
			self.EtiqContra.setText("Contraseña incorrecta")
	def funcion2(self):
		self.close()

if __name__ == "__main__":
        app = QtWidgets.QApplication(sys.argv)
        window = MyApp()
        app.exec_()
