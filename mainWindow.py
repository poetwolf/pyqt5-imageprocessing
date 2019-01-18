#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from mainWindowLayout import MainLayout

import cv2
import numpy as np
from scipy import ndimage

class MainWindow(QMainWindow, MainLayout):
    imagePaths = []
    originImages=[] 
    imageList = []  #二维的图像列表
    hideLayoutTag=-1

    def __init__(self,parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.signalSlots()

    #button与具体方法关联
    def signalSlots(self):
        self.importButton.clicked.connect(lambda : importImage(self))
        # self.histNormalizedButton.clicked.connect(lambda : layoutChange(self,1))
        # self.frequencyProcessButton.clicked.connect(lambda : layoutChange(self,2))
        self.morphologyProcessButton.clicked.connect(lambda : layoutChange(self,3))

def importImage(window):
    # 调用打开文件diglog
    fname, _ = QFileDialog.getOpenFileName(window, 'Open file', '.', 'Image Files(*.jpg *.bmp *.png *.jpeg *.rgb *.tif)')
    if fname!='':
        window.importImageEdit.setText(fname)
        window.imagePaths = []
        window.originImages = []
        window.imageList = []
        window.imagePaths.append(fname)
    if window.imagePaths!=[]:
        readIamge(window)
        resizeFromList(window, window.originImages)
        showImage(window)

def readIamge(window):
    window.originImages=[]
    for path in window.imagePaths:
        imgs=[]
        # img=cv2.imread(path)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        imgs.append(img)
        window.originImages.append(imgs)


def showImage(window,headers=[]):
    window.showImageView.clear()
    window.showImageView.setColumnCount(len(window.imageList[0]))
    window.showImageView.setRowCount(len(window.imageList))

    window.showImageView.setShowGrid(False)
    window.showImageView.setEditTriggers(QAbstractItemView.NoEditTriggers)
    window.showImageView.setHorizontalHeaderLabels(headers)
    for x in range(len(window.imageList[0])):
        for y in range(len(window.imageList)):
            imageView=QGraphicsView()
            imageView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            imageView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            img=window.imageList[y][x]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            width=img.shape[1]
            height=img.shape[0]

            window.showImageView.setColumnWidth(x, width)
            window.showImageView.setRowHeight(y, height)

            frame = QImage(img, width, height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            item = QGraphicsPixmapItem(pix)  # 创建像素图元
            scene = QGraphicsScene()  # 创建场景
            scene.addItem(item)
            imageView.setScene(scene)
            window.showImageView.setCellWidget(y, x, imageView)

def resizeFromList(window,imageList):
    width=800
    height=800
    window.imageList=[]
    for x_pos in range(len(imageList)):
        imgs=[]
        for img in imageList[x_pos]:

            image=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            imgs.append(image)
        window.imageList.append(imgs)
        print(len(window.imageList),len(window.imageList[0]))

def layoutChange(window,layoutTag):
    print(window.hideLayoutTag)
    if window.hideLayoutTag!=-1:
        deleteWidgets(window, window.hideLayoutTag)

    if layoutTag==1:
        window.rgbButton=QPushButton('RGB')
        window.hsiButton=QPushButton('HSV')
        window.hideLayout.addWidget(window.rgbButton)
        window.hideLayout.addWidget(window.hsiButton)

        #信号槽
        window.rgbButton.clicked.connect(lambda : histNormalized(window,'RGB'))
        window.hsiButton.clicked.connect(lambda : histNormalized(window,'HSV'))
    elif layoutTag==2:
        window.dftButton=QPushButton('快速傅里叶变换')
        window.gaussFilterButton=QPushButton('高斯低通滤波')
        window.hideLayout.addWidget(window.dftButton)
        window.hideLayout.addWidget(window.gaussFilterButton)

        #信号槽
        window.dftButton.clicked.connect(lambda : frequencyProcess(window,'DFT'))
        window.gaussFilterButton.clicked.connect(lambda : frequencyProcess(window,'Gauss Filter'))

    elif layoutTag==3:
        window.kernelLabel=QLabel('选择核大小:')
        window.kernelSelect=QComboBox()
        window.kernelSelect.addItems([str(i) for i in range(5,21)])
        window.excuteButton=QPushButton('执行')
        window.hideLayout.addWidget(window.kernelLabel)
        window.hideLayout.addWidget(window.kernelSelect)
        window.hideLayout.addWidget(window.excuteButton)

        #信号槽
        window.excuteButton.clicked.connect(lambda : morphologyProcess(window))

    window.cancelButton=QPushButton('取消')
    window.hideLayout.addWidget(window.cancelButton)
    window.cancelButton.clicked.connect(lambda: deleteWidgets(window, layoutTag))

    window.hideLayoutTag = layoutTag

def deleteWidgets(window,layoutMode):
    if layoutMode==1:
        window.rgbButton.deleteLater()
        window.hsiButton.deleteLater()

    elif layoutMode==2:
        window.dftButton.deleteLater()
        window.gaussFilterButton.deleteLater()

    elif layoutMode==3:
        window.kernelLabel.deleteLater()
        window.kernelSelect.deleteLater()
        window.excuteButton.deleteLater()

    window.cancelButton.deleteLater()
    window.hideLayoutTag=-1

def histNormalized(window,modelTag):
    imageList=[]
    if modelTag=='RGB':

        for img in window.originImages:
            imgs=[]
            b, g, r = cv2.split(img[0])
            b_equal = cv2.equalizeHist(b)
            g_equal = cv2.equalizeHist(g)
            r_equal = cv2.equalizeHist(r)
            bgrEquImage = cv2.merge([b_equal, g_equal, r_equal])
            imgs.extend([img[0],bgrEquImage])
            imageList.append(imgs)

        resizeFromList(window, imageList)
        showImage(window,['原图','均衡化后'])
    elif modelTag=='HSV':
        for img in window.originImages:
            imgs=[]
            hsvImage = cv2.cvtColor(img[0], cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsvImage)
            h_equal = cv2.equalizeHist(h)
            hEquImage = cv2.cvtColor(cv2.merge([h_equal, s, v]),cv2.COLOR_HSV2BGR)
            s_equal = cv2.equalizeHist(s)
            sEquImage=cv2.cvtColor(cv2.merge([h, s_equal, v]),cv2.COLOR_HSV2BGR)
            v_equal = cv2.equalizeHist(v)
            vEquImage = cv2.cvtColor(cv2.merge([h, s, v_equal]), cv2.COLOR_HSV2BGR)
            hsvEquImage = cv2.cvtColor(cv2.merge([h_equal, s_equal, v_equal]), cv2.COLOR_HSV2BGR)
            imgs.extend([img[0],hEquImage,sEquImage,vEquImage,hsvEquImage])
            imageList.append(imgs)

        resizeFromList(window, imageList)
        showImage(window,['RGB原图','H均衡化','S均衡化','V均衡化','HSV均衡化'])

def frequencyProcess(window,modelTag):
    imageList=[]
    if modelTag=='DFT':
        for images in window.originImages:
            for img in images:
                imgs=[]
                b,g,r=cv2.split(img)
                b_freImg,b_recImg=oneChannelDft(b)
                g_freImg, g_recImg = oneChannelDft(g)
                r_freImg, r_recImg = oneChannelDft(r)
                freImg=cv2.merge([b_freImg,g_freImg,r_freImg])
                recImg=cv2.merge([b_recImg,g_recImg,r_recImg])
                imgs.extend([img,freImg,recImg])
                imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window,['原图','傅里叶变换频谱','还原图'])
    elif modelTag=='Gauss Filter':
        for images in window.originImages:
            for img in images:
                imgs=[]
                imgs.append(img)
                b, g, r = cv2.split(img)
                width, height = b.shape
                nwidth = cv2.getOptimalDFTSize(width)
                nheigth = cv2.getOptimalDFTSize(height)
                for radius in [5,15,30,80,230]:
                    b_ilmg = oneChannelGaussLowFilter(b,nwidth,nheigth,radius)
                    g_ilmg = oneChannelGaussLowFilter(g,nwidth,nheigth,radius)
                    r_ilmg = oneChannelGaussLowFilter(r,nwidth,nheigth,radius)
                    ilmg=cv2.merge([b_ilmg,g_ilmg,r_ilmg])
                    imgs.append(ilmg)
                imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window,['原图','半径=5','半径=15','半径=30','半径=80','半径=230'])

def oneChannelDft(img):
    width, height = img.shape

    nwidth = cv2.getOptimalDFTSize(width)
    nheigth = cv2.getOptimalDFTSize(height)

    nimg = np.zeros((nwidth, nheigth))
    nimg[:width, :height] = img

    dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)

    ndft = dft[:width, :height]
    ndshift = np.fft.fftshift(ndft)
    magnitude = np.log(cv2.magnitude(ndshift[:, :, 0], ndshift[:, :, 1]))
    result = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255
    frequencyImg = result.astype('uint8')

    # dshift=np.fft.fftshift(dft)
    # idftShift=np.fft.ifftshift(dshift)
    ilmg = cv2.idft(dft)
    ilmg = cv2.magnitude(ilmg[:, :, 0], ilmg[:, :, 1])[:width, :height]
    ilmg = np.floor((ilmg - ilmg.min()) / (ilmg.max() - ilmg.min()) * 255)
    recoveredImg = ilmg.astype('uint8')

    return frequencyImg,recoveredImg

def oneChannelGaussLowFilter(img,nwidth,nheigth,radius):
    width, height = img.shape

    nimg = np.zeros((nwidth, nheigth))
    nimg[:width, :height] = img

    dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)


    dshift = np.fft.fftshift(dft)
    temp = cv2.magnitude(dshift[:, :, 0], dshift[:, :, 1])
    center = np.unravel_index(np.argmax(temp), temp.shape)
    gaussLow = creategaussLowFilter(nwidth, nheigth, center[0], center[1], radius)
    dshift[:, :, 0] = dshift[:, :, 0] * gaussLow
    dshift[:, :, 1] = dshift[:, :, 1] * gaussLow

    idftShift = np.fft.ifftshift(dshift)

    ilmg = cv2.idft(idftShift)
    ilmg = cv2.magnitude(ilmg[:, :, 0], ilmg[:, :, 1])[:width, :height]
    ilmg = (ilmg - ilmg.min()) / (ilmg.max() - ilmg.min()) * 255
    ilmg = ilmg.astype('uint8')

    return ilmg


def creategaussLowFilter(width, height, center_x, center_y, radius):
    gaussLowFilter = np.zeros((width, height), dtype='float32')
    Radius=2*np.power(radius,2.0)
    for x in range(width):
        for y in range(height):
            gaussLowFilter[x][y] = np.exp(
                -(np.power(x - center_x, 2.0) + np.power(y - center_y, 2.0)) / Radius)

    return gaussLowFilter

def morphologyProcess(window):
        imageList =[]
        for images in window.originImages:
            for img in images:
                imgs=[]
                # b,g,r=cv2.split()
                b,g,r=tuple(cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for image in cv2.split(img))
                ksize=int(window.kernelSelect.currentText())
                k=np.ones((ksize,ksize),np.uint8)
                binaryImg=cv2.merge([b,g,r])
                erodeImg=cv2.merge([cv2.erode(image,k) for image in [b,g,r]])
                dilateImage=cv2.merge([cv2.dilate(image,k) for image in [b,g,r]])
                openImage=cv2.merge([cv2.morphologyEx(image,cv2.MORPH_OPEN,k) for image in [b,g,r]])
                closeImage=cv2.merge([cv2.morphologyEx(image,cv2.MORPH_CLOSE,k) for image in [b,g,r]])
                imgs.extend([img,binaryImg,erodeImg,dilateImage,openImage,closeImage])
                imageList.append(imgs)

        resizeFromList(window, imageList)
        showImage(window,['原图','二值化','腐蚀','膨胀','开运算','闭运算'])


if __name__=='__main__':

    app=QApplication(sys.argv)
    mw=MainWindow()
    mw.show()
    sys.exit(app.exec_())
