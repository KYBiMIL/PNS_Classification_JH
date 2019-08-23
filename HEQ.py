import os, sys
import numpy as np
import random
import csv
import shutil
import cv2
from tkinter import Tk

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

'''
Coded by J.H
Date : 2019.5.28

'''
'''

def alpha_to_color(image, color=(255, 255, 255)):
	"""
	Set all fully transparent pixels of an RGBA image to the specified color.
    This is a very simple solution that might leave over some ugly edges, due
    to semi-transparent areas. You should use alpha_composite_with color instead.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    	""" 
    x = np.array(image)
    r, g, b, a = np.rollaxis(x, axis=-1)
    r[a == 0] = color[0]
    g[a == 0] = color[1]
    b[a == 0] = color[2] 
    x = np.dstack([r, g, b, a])
    return Image.fromarray(x, 'RGBA')
'''

def Get_Histogram(image, bins):

	histogram = np.zeros(bins)

	for pixel in image:
		histogram[pixel] += 1

	return histogram


def CumSum(a):
	a = iter(a)
	b = [next(a)]

	for i in a:
		b.append(b[-1] + i)

	return np.array(b)

	


def Histogram_Equalization3(filepath):

	img = Image.open(filepath)
	img_arr = np.asarray(img)

	flat = img_arr.flatten()
	hist = Get_Histogram(img_arr, 256)
	cdf = CumSum(hist)

	nj = (cdf - cdf.min()) * 255
	N = cdf.max() - cdf.min()
	cdf_m = nj / N
	cdf_m = cdf_m.astype('uint8')

	eq_img = cdf[flat]
	eq_img = np.reshape(eq_img, img_arr.shape)

	# return img, hist, cdf, cdf_m, eq_img
	return eq_img


def Histogram_Equalization2(filepath):

	img = cv2.imread(filepath, 0)

	cv_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
	hist, bins = np.histogram(img.flatten(), 256, [0, 256])

	cdf = hist.cumsum()
	cdf_m = np.ma.maked_equal(cdf, 0)
	cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
	cdf = np.ma.filled(cdf_m, 0).astype('uing8')
	
	eq_img = cdf[img]

	# return img, hist, cdf, cdf_m, eq_img
	return eq_img


def Histogram_Equalization(filepath):
	
	img = cv2.imread(filepath, 0)
	eq_img = cv2.equalizeHist(img)
	
	return eq_img


def Adaptive_Histogram_Equalization(filepath, n_tiles):

	img = cv2.imread(filepath, 0)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(n_tiles, n_tiles))
	CI = clahe.apply(img)
	
	return img, CI





'''
f1 = "/data/PNS/test/data/002186_180601_1_1_R.jpg"
f2 = "/data/PNS/test/data/003459_171128_0_0_R.jpg"
f3 = "/data/PNS/test/data/004611_170406_0_0_R.jpg"
f4 = "/data/PNS/test/data/011070_170619_1_1_R.png"
f5 = "/data/PNS/test/data/013812_180601_1_2_R.png"

F = f5
'''
"""

## Histogram Equalization

img = Image.open(F)
plt.imshow(img, cmap='gray')


img_arr = np.asarray(img)

flat = img_arr.flatten()
plt.hist(flat, bins=256)

hist = Get_Histogram(img_arr, 256)

cdf = CumSum(hist)
plt.plot(cdf)

nj = (cdf - cdf.min()) * 255
N = cdf.max() - cdf.min()

cdf = nj / N
cdf = cdf.astype('uint8')
plt.plot(cdf)

new_img = cdf[flat]
new_img = np.reshape(new_img, img_arr.shape)

fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

cv2.imshow("Image", img_arr)
cv2.imshow("Equalized Image", new_img)


## Adaptive Histogram Equalization

image = cv2.imread(F, 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
CI = clahe.apply(image)
cv2.imshow("CLAHE", CI)
cv2.waitKey(10000)

"""




