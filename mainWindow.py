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
        # 初始化一个img的ndarray, 用于存储图像
        self.img = np.ndarray(())

    #button与具体方法关联
    def signalSlots(self):
        #文件按钮相关方法
        #打开
        self.openAct.triggered.connect(lambda : importImage(self))
        #保存
        self.saveAct.triggered.connect(lambda : importImage(self))
        #退出
        self.exitAct.triggered.connect(lambda : importImage(self))

        #编辑按钮相关方法
        #放大
        self.largeAct.triggered.connect(lambda : largeImage(self))
        #缩小
        self.smallAct.triggered.connect(lambda : smallImage(self))
        #灰度
        self.grayAct.triggered.connect(lambda : grayImage(self))
        #亮度
        self.brightAct.triggered.connect(lambda : importImage(self))
        #旋转
        self.rotateAct.triggered.connect(lambda : rotateImage(self))
        #截图
        self.screenshotAct.triggered.connect(lambda : screenshotImage(self))

        #变换按钮相关方法
        #傅里叶变换
        self.change1Act.triggered.connect(lambda : importImage(self))
        #离散余弦变换
        self.change2Act.triggered.connect(lambda : importImage(self))
        #Radom变换
        self.change3Act.triggered.connect(lambda : importImage(self))

        #噪声按钮相关方法
        #高斯噪声
        self.noise1Act.triggered.connect(lambda : importImage(self))
        #椒盐噪声
        self.noise2Act.triggered.connect(lambda : importImage(self))
        #斑点噪声
        self.noise3Act.triggered.connect(lambda : importImage(self))
        #泊松噪声
        self.noise4Act.triggered.connect(lambda : importImage(self))

        #滤波按钮相关方法
        #高通滤波
        self.smoothing1Act.triggered.connect(lambda : importImage(self))
        #低通滤波
        self.smoothing2Act.triggered.connect(lambda : importImage(self))
        #平滑滤波
        self.smoothing3Act.triggered.connect(lambda : importImage(self))
        #锐化滤波
        self.smoothing4Act.triggered.connect(lambda : importImage(self))

        #直方图统计按钮相关方法
        #R直方图
        self.smoothing1Act.triggered.connect(lambda : importImage(self))
        #G直方图
        self.smoothing2Act.triggered.connect(lambda : importImage(self))
        #B直方图
        self.smoothing3Act.triggered.connect(lambda : importImage(self))

        #图像增强按钮相关方法
        #伪彩色增强
        self.enhance1Act.triggered.connect(lambda : importImage(self))
        #真彩色增强
        self.enhance2Act.triggered.connect(lambda : importImage(self))
        #直方图均衡
        self.enhance3Act.triggered.connect(lambda : histNormalized(self))
        #NTSC颜色模型
        self.enhance4Act.triggered.connect(lambda : importImage(self))
        #YCbCr颜色模型
        self.enhance5Act.triggered.connect(lambda : importImage(self))
        #HSV颜色模型
        self.enhance6Act.triggered.connect(lambda : importImage(self))

        #阈值分割方法
        self.threButton.clicked.connect(lambda : layoutChange(self,3))
        #形态学处理方法
        self.morphologyProcessButton.clicked.connect(lambda : layoutChange(self))
        #特征提取方法
        self.featureButton.clicked.connect(lambda : layoutChange(self,3))
        #图像分类与识别方法
        self.imgButton.clicked.connect(lambda : layoutChange(self,3))

#编辑按钮相关方法
#放大
def largeImage(window):
    imageList=[]
    for img in window.originImages:
        imgs=[]
        img_info=img[0].shape
        image_height=img_info[0]
        image_weight=img_info[1]    
        dstHeight=int(2*image_height)
        dstWeight=int(2*image_weight)
        result=cv2.resize(img[0],(dstHeight,dstWeight))
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','放大后'])
#缩小
def smallImage(window):
    imageList=[]
    for img in window.originImages:
        imgs=[]
        img_info=img[0].shape
        image_height=img_info[0]
        image_weight=img_info[1]    
        dstHeight=int(0.5*image_height)
        dstWeight=int(0.5*image_weight)
        result=cv2.resize(img[0],(dstHeight,dstWeight))
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','缩小后'])
#灰度
def grayImage(window):
    imageList=[]
    for img in window.originImages:
        imgs=[]
        b = cv2.CreateImage(cv2.GetSize(img[0]), img[0].depth, 1)
        g = cv2.CloneImage(b)
        r = cv2.CloneImage(b)
    
        
        result = cv2.Split(img[0], b, g, r, None)
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','灰度处理后'])
#旋转
def rotateImage(window):
    imageList=[]
    for img in window.originImages:
        imgs=[]
        img_info=img[0].shape
        image_height=img_info[0]
        image_weight=img_info[1]
        mat_rotate=cv2.getRotationMatrix2D((image_height*0.5,image_weight*0.5),90,1)    #center angle 3scale
        result=cv2.warpAffine(img[0],mat_rotate,(image_height,image_weight))
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','旋转后'])
#截图
def screenshotImage(window):
    imageList=[]
    for img in window.originImages:
        imgs=[]
        result = img[0][70:170, 440:540]
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','截图后'])

#图像增强按钮相关方法
#直方图均衡
def histNormalized(window):
    imageList=[]

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

#打开图像
def importImage(window):
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

#显示图像
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
    width=600
    height=500
    window.imageList=[]
    for x_pos in range(len(imageList)):
        imgs=[]
        for img in imageList[x_pos]:

            image=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            imgs.append(image)
        window.imageList.append(imgs)
        print(len(window.imageList),len(window.imageList[0]))

if __name__=='__main__':

    app=QApplication(sys.argv)
    mw=MainWindow()
    mw.show()
    sys.exit(app.exec_())
