import os
import sys
import numpy as np
import cv2
import glob
from PIL import Image
import PIL




def Juhui_G_RL(path):

	img_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in files if all(s in f for s in ['.jpg'])]
	img_list.sort()

	side = []
	f_side = []

	for i, l in enumerate(img_list):
		img = cv2.imread(l, 0)
		h, w = img.shape

		half_h = int(h/2)
		half_w = int(w/2)

		L_img = img[:, :half_w]
		R_img = img[:, half_w:]

		L_mean = L_img.mean()
		R_mean = R_img.mean()

		if L_mean > R_mean:
			side.append('L')
			f_side.append(l + "	L")
		else:
			side.append('R')
			f_side.append(l + "	R")

	f = open("./g_glaucoma_f_side.txt", 'w')
	s = open("./g_glaucoma_side.txt", 'w')

	for lr in f_side:
		f.write(lr + '\n')
	for flr in side:
		s.write(flr + '\n')
	f.close()
	s.close()

path = "/data/Fundus Image/g_glaucoma"
Juhui_G_RL(path)
