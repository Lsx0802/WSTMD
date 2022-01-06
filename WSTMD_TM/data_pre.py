# coding=utf-8
# coding=utf-8
import os
import numpy as np
import cv2 as cv
import math
from tqdm import tqdm
import random


# 无齿痕切割，沿边缘切割

def gleftItem(rgbImage, start_h, h, w, box_size, beishu=32):
    C_r = 0
    C_c = 0
    for row in range(start_h, h):
        b_r = False
        for col in range(w):
            pv0 = rgbImage[col, row, 0]
            pv1 = rgbImage[col, row, 1]
            pv2 = rgbImage[col, row, 2]
            if pv0 > 0 or pv1 > 0 or pv2 > 0:  # 从舌体中段开始向下，从左到右判断像素体是否为舌体，判断为舌体边缘时停止
                C_r = row   #y
                C_c = col   #x
                b_r = True
                break
        if b_r == True:
            # print('stop----------------------')
            break
    if C_c != 0 or C_r != 0:
        if C_c < box_size[0] / 2:
            if C_r > h - box_size[1] / 2:
                return 0, round((h - box_size[1]) / beishu), box_size[0] / beishu, box_size[1] / beishu, C_r
            else:
                return 0, round((C_r - box_size[1] / 2) / beishu), box_size[0] / beishu, box_size[1] / beishu, C_r
        else:
            if C_r > h - box_size[1] / 2:
                return round((C_c - box_size[0] / 2) / beishu), round((h - box_size[1]) / beishu), box_size[0] / beishu, \
                       box_size[1] / beishu, C_r
            else:
                return round((C_c - box_size[0] / 2) / beishu), round((C_r - box_size[1] / 2) / beishu), box_size[
                    0] / beishu, box_size[1] / beishu, C_r
    else:
        return 0, 0, 0, 0, 0


def genleftBox(img, xywh, box_size, N, step, Imagesize):
    rgbImage = cv.imread(img, cv.IMREAD_COLOR)
    rgbImage = cv.resize(rgbImage, (Imagesize, Imagesize))
    w, h, c = rgbImage.shape
    # 找到第一个像素点
    x, y, w_, h_, C_r = gleftItem(rgbImage, 0, h, w, box_size)
    xywh.append(x)
    xywh.append(y)
    xywh.append(w_)
    xywh.append(h_)
    for i in range(1, int(N / 2)):
        x, y, w_, h_, C_r = gleftItem(rgbImage, C_r + step * i, h, w, box_size)
        if x == 0 and y == 0 and w_ == 0 and h_ == 0:
            print('Warning')
            break
        xywh.append(x)
        xywh.append(y)
        xywh.append(w_)
        xywh.append(h_)


def gRightItem(rgbImage, start_h, h, w,box_size,beishu=32):
    C_r = 0
    C_c = 0
    for row in range(start_h, h):
        b_r = False
        for col in range(w - 1, -1, -1):
            pv0 = rgbImage[col, row, 0]
            pv1 = rgbImage[col, row, 1]
            pv2 = rgbImage[col, row, 2]
            if pv0 > 0 or pv1 > 0 or pv2 > 0:
                # print(row, col)
                C_r = row
                C_c = col
                b_r = True
                break
        if b_r == True:
            # print('stop----------------------')
            break
    if C_c != 0 or C_r != 0:
        if C_c > w - box_size[0]/2:
            if C_r > h - box_size[1]/2:
                return round((w - box_size[0]) / beishu), round((h - box_size[1]) / beishu), box_size[0] / beishu, box_size[1] / beishu, C_r
            else:
                return round((w - box_size[0]) / beishu), round((C_r - box_size[1]/2) / beishu), box_size[0] / beishu, box_size[1] / beishu, C_r
        else:
            if C_r > h - box_size[1]/2:
                return round((C_c - box_size[0]/2) / beishu), round((h - box_size[1]) / beishu), box_size[0] / beishu, box_size[1] / beishu, C_r
            else:
                return round((C_c - box_size[0]/2) / beishu), round((C_r - box_size[1]/2) / beishu), box_size[0] / beishu, box_size[1] / beishu, C_r
    else:
        return 0, 0, 0, 0, 0


def genrightBox(img, xywh,box_size, N, step, Imagesize):
    rgbImage = cv.imread(img, cv.IMREAD_COLOR)
    rgbImage = cv.resize(rgbImage, (Imagesize, Imagesize))
    w, h, c = rgbImage.shape
    # 找到第一个像素点
    x, y, w_, h_, C_r = gRightItem(rgbImage, 0, h, w,box_size)
    xywh.append(x)
    xywh.append(y)
    xywh.append(w_)
    xywh.append(h_)
    for i in range(1, int(N / 2)):
        x, y, w_, h_, C_r = gRightItem(rgbImage, C_r + step * i, h, w,box_size)
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
    genleftBox(img, xywh, box_size, N=20, step=4, Imagesize=224)
    genrightBox(img, xywh, box_size, N=20, step=4, Imagesize=224)

    return xywh


def scan_files(path):
    box_size = [32,32]
    patient = os.listdir(path)
    f = open('ssw_4.txt', 'w')
    minlen = 10000
    for img in tqdm(patient):
        xywh = []
        img_path = os.path.join(path, img)
        genleftBox(img_path, xywh, box_size, N=20, step=4, Imagesize=224)
        f.write(str(img) + " " + " ".join(str(int(i)) for i in np.array(xywh)))
        xywh = []
        genrightBox(img_path, xywh, box_size, N=20, step=4, Imagesize=224)
        print(len(xywh))
        if len(xywh) < minlen:
            minlen = len(xywh)
        f.write(' ' + ' '.join(str(int(i)) for i in np.array(xywh)) + '\n')
    f.close()
    print(minlen / 4)

# scan_files('data/img')
