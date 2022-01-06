# coding=utf-8
# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import PIL.Image as Image
import numpy as np
import time
from selectivesearch.selectivesearch import selective_search

def main():
    candidates=[1 ,0 ,2 ,2, 1, 0, 2, 2, 0, 1 ,2 ,2, 0 ,2 ,2, 2, 0 ,3 ,2, 2, 1 ,0 ,2 ,2 ,3 ,0, 2 ,2, 3 ,1 ,2 ,2, 3, 2 ,2 ,2, 3, 3, 2, 2]
    # loading astronaut image

    # img = skimage.data.astronaut()
    before = time.time()
    img=Image.open('3.png').convert("RGB")
    img = np.array(img)

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for i in candidates:
        x=i
        y=i+1
        w=i+2
        h=i+3
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    after=time.time()
    print('total time = '+str((after-before)/60)+' min')
    plt.show()

if __name__ == "__main__":
    main()
