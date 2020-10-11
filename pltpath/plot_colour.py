# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:42:48 2020

@author: alekh
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def get_images():
    fname = '0_img_ttjpen.png'
    full_img = Image.open(fname)
    full_img = np.asarray(full_img)
    img_list = []
    startx = 280 + 160 * 5
    for i in range(0, 6, 1):
        img = full_img[i * 160 + 20 : i * 160 + 20 + 150, startx : startx + 150]
        #print(startx , startx + 150, i * 160 + 20, i * 160 + 20 + 150)
        img = np.where(img == 0, 0, i)
        img_list.append(img)
    return img_list

def load_images():
    fnameb = 'out{0}.png'
    
    img_list = []
    
    for i in range(0, 7, 1):
        fname = fnameb.format(i)
        print(fname, i)
        img = Image.open(fname)
        img = np.asarray(img)
        if i > 0:
            img_list.append(np.where(img - sumimg == 0, 0, 255) * i) # - img_list[-1])
            sumimg = sumimg + img
            sumimg = np.where(sumimg == 0, 0, 255)
        else:
            img_list.append(img)
            sumimg = img
    return img_list

def plot_colour(ims):
    
    norm = matplotlib.colors.BoundaryNorm(list(range(5)), 5)
    plt.imshow(ims, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    
#im = get_images()
im = load_images()

im = np.array(im)
# ims = np.sum(im, axis=0, dtype=np.float32)
imm = np.sum(im, axis=0)
imm = np.clip(imm, 0, 1500)
imm = np.where(imm == 1500, 0, imm)

plot_colour(imm)


