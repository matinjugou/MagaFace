# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.infoTab = QtWidgets.QWidget()
        self.infoTab.setObjectName("infoTab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.infoTab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.infoTable = QtWidgets.QTableWidget(self.infoTab)
        self.infoTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.infoTable.setObjectName("infoTable")
        self.infoTable.setColumnCount(2)
        self.infoTable.setRowCount(8)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(2, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(3, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(4, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(5, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(6, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.infoTable.setItem(7, 1, item)
        self.infoTable.horizontalHeader().setVisible(True)
        self.infoTable.horizontalHeader().setDefaultSectionSize(100)
        self.infoTable.horizontalHeader().setStretchLastSection(True)
        self.infoTable.verticalHeader().setVisible(False)
        self.verticalLayout_2.addWidget(self.infoTable)
        self.tabWidget.addTab(self.infoTab, "")
        self.widget = QtWidgets.QWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.video = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video.sizePolicy().hasHeightForWidth())
        self.video.setSizePolicy(sizePolicy)
        self.video.setMinimumSize(QtCore.QSize(400, 400))
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setObjectName("video")
        self.verticalLayout.addWidget(self.video)
        self.playControls = QtWidgets.QHBoxLayout()
        self.playControls.setObjectName("playControls")
        self.playButton = QtWidgets.QPushButton(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.playButton.sizePolicy().hasHeightForWidth())
        self.playButton.setSizePolicy(sizePolicy)
        self.playButton.setMinimumSize(QtCore.QSize(30, 0))
        self.playButton.setMaximumSize(QtCore.QSize(30, 16777215))
        self.playButton.setObjectName("playButton")
        self.playControls.addWidget(self.playButton)
        self.playSlider = QtWidgets.QSlider(self.widget)
        self.playSlider.setEnabled(False)
        self.playSlider.setOrientation(QtCore.Qt.Horizontal)
        self.playSlider.setObjectName("playSlider")
        self.playControls.addWidget(self.playSlider)
        self.progressText = QtWidgets.QLabel(self.widget)
        self.progressText.setObjectName("progressText")
        self.playControls.addWidget(self.progressText)
        self.verticalLayout.addLayout(self.playControls)
        self.horizontalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1015, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        MainWindow.setMenuBar(self.menubar)
        self.openFile = QtWidgets.QAction(MainWindow)
        self.openFile.setObjectName("openFile")
        self.openStream = QtWidgets.QAction(MainWindow)
        self.openStream.setObjectName("openStream")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.showSidebar = QtWidgets.QAction(MainWindow)
        self.showSidebar.setCheckable(True)
        self.showSidebar.setChecked(True)
        self.showSidebar.setObjectName("showSidebar")
        self.menuFile.addAction(self.openFile)
        self.menuFile.addAction(self.openStream)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuView.addAction(self.showSidebar)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        item = self.infoTable.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "File"))
        item = self.infoTable.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Frame"))
        item = self.infoTable.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Time(ms)"))
        item = self.infoTable.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "Progress"))
        item = self.infoTable.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "Width"))
        item = self.infoTable.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "Height"))
        item = self.infoTable.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "Frame Rate"))
        item = self.infoTable.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "Total Frames"))
        item = self.infoTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Property"))
        item = self.infoTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Value"))
        __sortingEnabled = self.infoTable.isSortingEnabled()
        self.infoTable.setSortingEnabled(False)
        item = self.infoTable.item(0, 0)
        item.setText(_translate("MainWindow", "File"))
        item = self.infoTable.item(0, 1)
        item.setText(_translate("MainWindow", "None"))
        item = self.infoTable.item(1, 0)
        item.setText(_translate("MainWindow", "Frame"))
        item = self.infoTable.item(1, 1)
        item.setText(_translate("MainWindow", "0"))
        item = self.infoTable.item(2, 0)
        item.setText(_translate("MainWindow", "Time(ms)"))
        item = self.infoTable.item(2, 1)
        item.setText(_translate("MainWindow", "0"))
        item = self.infoTable.item(3, 0)
        item.setText(_translate("MainWindow", "Progress"))
        item = self.infoTable.item(3, 1)
        item.setText(_translate("MainWindow", "0.00%"))
        item = self.infoTable.item(4, 0)
        item.setText(_translate("MainWindow", "Width"))
        item = self.infoTable.item(4, 1)
        item.setText(_translate("MainWindow", "0"))
        item = self.infoTable.item(5, 0)
        item.setText(_translate("MainWindow", "Height"))
        item = self.infoTable.item(5, 1)
        item.setText(_translate("MainWindow", "0"))
        item = self.infoTable.item(6, 0)
        item.setText(_translate("MainWindow", "Frame Rate"))
        item = self.infoTable.item(6, 1)
        item.setText(_translate("MainWindow", "0"))
        item = self.infoTable.item(7, 0)
        item.setText(_translate("MainWindow", "Total Frames"))
        item = self.infoTable.item(7, 1)
        item.setText(_translate("MainWindow", "0"))
        self.infoTable.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.infoTab), _translate("MainWindow", "Video Info"))
        self.video.setText(_translate("MainWindow", "No video selected"))
        self.playButton.setText(_translate("MainWindow", "⏵"))
        self.progressText.setText(_translate("MainWindow", "00:00/∞"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.openFile.setText(_translate("MainWindow", "&Open Video..."))
        self.openFile.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.openStream.setText(_translate("MainWindow", "Open &RTSP"))
        self.openStream.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExit.setText(_translate("MainWindow", "&Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.showSidebar.setText(_translate("MainWindow", "Show &Sidebar"))
        self.showSidebar.setShortcut(_translate("MainWindow", "Ctrl+V"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())