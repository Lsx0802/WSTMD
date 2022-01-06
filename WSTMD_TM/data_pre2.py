# coding=utf-8
# coding=utf-8
import os
import numpy as np
import cv2 as cv
import math
from tqdm import tqdm
import random
import time

# 无齿痕切割，沿边缘切割

def gleftItem(rgbImage, start_h, h, w, box_size):
    rgbImage = cv.cvtColor(rgbImage, cv.COLOR_BGR2GRAY)
    C_r = 0
    C_c = 0
    for row in range(start_h, h):
        b_r = False
        for col in range(w):
            if rgbImage[row][col]>0:  # 从舌体中段开始向下，从左到右判断像素体是否为舌体，判断为舌体边缘时停止
                C_r = row   #y
                C_c = col   #x
                # print(rgbImage[col][row])
                # print(col)
                # print(row)
                b_r = True
                break
        if b_r == True:
            # print('stop----------------------')
            break
    if C_c != 0 or C_r != 0:
        if C_c < (box_size[0] / 2):
            if C_r > (h - box_size[1] / 2):
                return 0, ((h - box_size[1]) ), box_size[0] , box_size[1] , C_r
            else:
                return 0, ((C_r - box_size[1] / 2) ), box_size[0] , box_size[1] , C_r
        else:
            if C_r > (h - box_size[1] / 2):
                return ((C_c - box_size[0] / 2) ), ((h - box_size[1]) ), box_size[0] , \
                       box_size[1] , C_r
            else:
                return ((C_c - box_size[0] / 2) ), ((C_r - box_size[1] / 2) ), box_size[
                    0] , box_size[1] , C_r
    else:
        return 0, 0, 0, 0, 0


def genleftBox(img, xywh, box_size, N, step, Imagesize):
    rgbImage = cv.imread(img, cv.IMREAD_COLOR)
    rgbImage = cv.resize(rgbImage, (Imagesize, Imagesize))
    w, h, c = rgbImage.shape
    # 找到第一个像素点
    x, y, w_, h_, C_r = gleftItem(rgbImage, 0, h, w, box_size)
    # xywh.append(x)
    # xywh.append(y)
    # xywh.append(w_)
    # xywh.append(h_)
    for i in range(1, int(N / 2)):
        x, y, w_, h_, C_r = gleftItem(rgbImage, C_r + step , h, w, box_size)
        if x == 0 and y == 0 and w_ == 0 and h_ == 0:
            print('Warning')
            break
        xywh.append(x)
        xywh.append(y)
        xywh.append(w_)
        xywh.append(h_)


def gRightItem(rgbImage, start_h, h, w,box_size):
    rgbImage = cv.cvtColor(rgbImage, cv.COLOR_BGR2GRAY)
    C_r = 0
    C_c = 0
    for row in range(start_h, h):
        b_r = False
        for col in range(w - 1, -1, -1):
            if rgbImage[row][col]>0:
                # print(row, col)
                C_r = row
                C_c = col
                b_r = True
                break
        if b_r == True:
            # print('stop----------------------')
            break
    if C_c != 0 or C_r != 0:
        if C_c > (w - box_size[0]/2):
            if C_r > (h - box_size[1]/2):
                return ((w - box_size[0]) ), ((h - box_size[1]) ), box_size[0] , box_size[1] , C_r
            else:
                return ((w - box_size[0]) ), ((C_r - box_size[1]/2) ), box_size[0] , box_size[1] , C_r
        else:
            if C_r >( h - box_size[1]/2):
                return ((C_c - box_size[0]/2) ), ((h - box_size[1]) ), box_size[0] , box_size[1] , C_r
            else:
                return ((C_c - box_size[0]/2) ), ((C_r - box_size[1]/2) ), box_size[0] , box_size[1] , C_r
    else:
        return 0, 0, 0, 0, 0


def genrightBox(img, xywh,box_size, N, step, Imagesize):
    rgbImage = cv.imread(img, cv.IMREAD_COLOR)
    rgbImage = cv.resize(rgbImage, (Imagesize, Imagesize))
    w, h, c = rgbImage.shape
    # 找到第一个像素点
    x, y, w_, h_, C_r = gRightItem(rgbImage, 0, h, w,box_size)
    # xywh.append(x)
    # xywh.append(y)
    # xywh.append(w_)
    # xywh.append(h_)
    for i in range(1, int(N / 2)):
        x, y, w_, h_, C_r = gRightItem(rgbImage, C_r + step , h, w,box_size)
        if x == 0 and y == 0 and w_ == 0 and h_ == 0:
            print('Warning')
            break
        xywh.append(x)
        xywh.append(y)
        xywh.append(w_)
        xywh.append(h_)


def box11(img):
    box_size = [32,32]
    xywh = []
    genleftBox(img, xywh, box_size, N=20, step=24, Imagesize=224)
    genrightBox(img, xywh, box_size, N=20, step=24, Imagesize=224)
    return xywh


def main1(path):
    box_size = [32,32]
    patient = os.listdir(path)
    f = open('ssw_5.txt', 'w')
    minlen = 10000
    for img in tqdm(patient):
        xywh = []
        img_path = os.path.join(path, img)
        genleftBox(img_path, xywh, box_size, N=20, step=24, Imagesize=224)
        f.write(str(img) + " " + " ".join(str(int(i)) for i in np.array(xywh)))
        xywh = []
        genrightBox(img_path, xywh, box_size, N=20, step=24, Imagesize=224)
        print(len(xywh))
        if len(xywh) < minlen:
            minlen = len(xywh)
        f.write(' ' + ' '.join(str(int(i)) for i in np.array(xywh)) + '\n')
    f.close()
    print(minlen / 4)

def main2(path):
    cul = os.listdir(path)
    before = time.time()
    for patient in tqdm(cul):
        path_cul = os.path.join(path, patient)
        path_cul = os.path.join(path_cul, 'tong.png')
        box11(path_cul)

    after = time.time()
    print(str((after-before)/10))

def main3(path):
    box11(path)
if __name__ == '__main__':
    # main1('data/img')
    # main2(r'C:\Users\hello\PycharmProjects\tongue\tongue_resnet_WSDDN\cul_time')
    main3(r'C:\Users\hello\PycharmProjects\tongue\tongue_resnet_WSDDN\data\img\140829091953.png')
