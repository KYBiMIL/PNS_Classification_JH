import os
import sys
import numpy as np
import cv2
import glob
from PIL import Image
import PIL
import pandas as pd





def Make_Directory(path):

	if not os.path.exists(path):
		os.makedirs(path)

	return True



def GetUniqueSize(path):

	imList = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in files if all(s in f for s in [".jpg"])]
	imList.sort()

	hwDict = {}
	for i, l in enumerate(imList):

		img = cv2.imread(l, 3)
		h, w, c = img.shape

		if (h, w, c) in hwDict:
			num = hwDict[(h, w, c)]
			hwDict.update({(h, w, c): (num+1)})	
		else:
			hwDict.update({(h, w, c): 1})
		

	print(hwDict)
	return hwDict
		

def ResizeSave(path, hwDict):
	
	n = 0
	lh, lw, lc = (0, 0, 0)
	for i, d in enumerate(hwDict):
		if hwDict[d] > n:
			lh, lw, lc = d
			n = hwDict[d]


	imList = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in files if all(s in f for s in [".jpg"])]
	imList.sort()

	for i, l in enumerate(imList):

		img = cv2.imread(l, 3)

		if not img.shape == (lh, lw, lc):
			oh, ow, oc = img.shape
			res = cv2.resize(img, (lw, lh))

			savedir = os.path.join(path, "resizeTest", "from" + str(oh) + str(ow))
			Make_Directory(savedir)
			savefn = l.split('/')[-1]

			cv2.imwrite(os.path.join(savedir, savefn), res)
	return True




#R1 = "/data/Fundus Image/AnonymizedData/Round1/6.광각_RVO"
#R2 = "/data/Fundus Image/AnonymizedData/Round2/6.광각_RVO"

#path = R2

#hwDict = GetUniqueSize(path)
#ResizeSave(path, hwDict)




'''
1.일반_정상
2.일반_녹내장
3.광각_정상
4.광각_AMD
5.광각_DMR
6.광각_RVO
'''



def JuhuiQuestGC(path):
	
	imList = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in files if all(s in f for s in [".jpg"])]
	imList.sort()

	fn = os.path.join(path, 'list.txt')
	with open(fn, 'w') as f:
		for l in imList:
			f.write("%s\n" % l.split('/')[-1])
		f.close()

	return True


#path = "/data/Fundus Image/AnonymizedData/Round2/1.일반_정상/resizeTest"
#JuhuiQuestGC(path)



def JuhuiQuestGC2(path = "/data/Fundus Image/AnonymizedData/Round1/2.일반_녹내장"):

	h, w = (2136, 3216)
	
	imList = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in files if all(s in f for s in [".jpg"])]
	imList.sort()

	for i, l in enumerate(imList):
		img = cv2.imread(l, 3)

		if not img.shape == (h, w, 3):
			oh, ow, _ = img.shape
			res = cv2.resize(img, (w, h))

			savedir = os.path.join(path, "resizeTest2", "from" + str(oh)+str(ow))
			Make_Directory(savedir)
			savepath = os.path.join(savedir, l.split('/')[-1])

			cv2.imwrite(savepath, res)

	return True

#JuhuiQuestGC2()

