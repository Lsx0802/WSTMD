# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import cv2
import PIL.Image as Image
import numpy as np
import time
import os
from tqdm import tqdm
from selectivesearch.selectivesearch import selective_search

def main():
    # path=r'C:\Users\hello\PycharmProjects\tongue\tongue_resnet_WSDDN\cul_time'
    # cul=os.listdir(path)
    # before = time.time()
    # for patient in tqdm(cul):
    #     path_cul=os.path.join(path,patient)
    #     path_cul = os.path.join(path_cul,'tong.png')
    # loading astronaut image

    # img = skimage.data.astronaut()
    path_cul=r'C:\Users\hello\PycharmProjects\tongue\tongue_resnet_WSDDN\data\img\140829091953.png'

    # rgbImage = cv2.imread(path_cul, cv2.IMREAD_COLOR)
    # img = cv2.resize(rgbImage, (224, 224))

    img=Image.open(path_cul).convert("RGB")
    # img=img.resize([224,224])
    img = np.array(img)

    # perform selective search
    img_lbl, regions = selective_search(
        img, scale=1, sigma=0.8, min_size=50)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        # print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='yellow', linewidth=1)
        ax.add_patch(rect)
    plt.show()
    # after = time.time()
    # return (after-before)/10

if __name__ == "__main__":
    # times=main()
    # print('total time = '+str((times)))

    main()