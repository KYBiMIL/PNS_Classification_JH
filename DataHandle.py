import os, sys
import numpy as np
import random
import csv
import shutil
from skimage import io
from PIL import Image
# import scipy
import matplotlib.pyplot as plt
import cv2
import pydicom	#, png
import scipy.misc
# from Preprocessing import *





'''
Coded by J.H
Date : 2019.08.13

'''

def Make_Directory(path):

	if not os.path.exists(path):
		os.makedirs(path)

	return True


def Crop_ROI(img, upb = 350, lwb = 950):


	cr_img = img[upb:lwb, ]

	return cr_img		# 600xc



def Normalization(img):

	i_mean = img.mean()
	i_std = img.std()

	# normalized_img = (img - i_mean) / i_std

	epsilon = 1e-6
	normalized_img = (img - i_mean) / (i_std + epsilon)


	return normalized_img




def Shuffle_List(p_list, n = 3):

	for i in range (3):	# 0, 1, 2
		random.shuffle(p_list)

	return p_list





def ReadFileLines(filename):

	Lines = []

	f = open(filename, 'r')

	for line in f:
		if not line.startswith('#'):
			Lines.append(line.strip())
	f.close()

	return Lines


def RenameID(list):

	nid = 8
	for i, l in enumerate(list):
		if len(l) == 1:
			list[i] = '0000000' + l		
		elif len(l) == 2:
			list[i] = '000000' + l
		elif len(l) == 3:
			list[i] = '00000' + l
		elif len(l) == 4:
			list[i] = '0000' + l
		elif len(l) == 5:
			list[i] = '000' + l
		elif len(l) == 6:
			list[i] = '00' + l
		elif len(l) == 7:
			list[i] = '0' + l
		else:
			print(str(i) + '-th ' + l + " got the wrong way!\n\n\n")

	return list

			

'''
fd_list = 'Data_List'
fd_WXR = '../KJY_PNS_Radiographs'
fd_CT = '../OMUCT/KJY_PNS'
savepath_WXR = '../PNS_WXR'
savepath_WXRwCT = '../PNS_WXRwCT'

prefix_CT = 'CT_'
prefix_WXR = 'WXR_'
prefix_WXRwCT = 'WXRwCT_'
'''

def DataArrangement_WXR(fd_list = 'Data_List', fd_WXR = '../KJY_PNS_Radiographs_20190814', savepath_WXR = '../PNS_WXR'):

	prefix_WXR = 'WXR_'
	file_ext = '.txt'
	dcm_ext = '.dcm'
	jpg_ext = '.jpg'

	Year_WXR = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

	for i, y_WXR in enumerate(Year_WXR):
		Make_Directory(os.path.join(savepath_WXR, prefix_WXR + y_WXR))

		fn = os.path.join(fd_list, prefix_WXR + y_WXR + file_ext)
		PID = ReadFileLines(fn)
		LPID = RenameID(PID)			# patinets, 00000000, in the year 

		dataDir = os.path.join(fd_WXR, y_WXR)

	
		all_WXR_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(dataDir) for f in files if all(s in f for s in ['.dcm'])]
		all_WXR_list.sort()

		WXR_ID = []
		WXR_list = []
		for j, wxr in enumerate(all_WXR_list):
			_, tmp1 = wxr.split('31/')
			tmp2, dcmfile = tmp1.split('/')
			ID, data, time, CR = tmp2.split('_')

			if set([ID])&set(LPID):
				
				if not set([ID])&set(WXR_ID):
					WXR_ID.append(ID)
					WXR_list.append(wxr)
				else:
					__, _tmp1 = WXR_list[-1].split('31/')
					_tmp2, _dcmfile = _tmp1.split('/')
					_ID, _data, _time, _CR = _tmp2.split('_')

					data_wxr = int(data + time)
					DATA_WXR_list = int(_data + _time)

					if data_wxr < DATA_WXR_list:
						WXR_list[-1] = wxr


		for k, WXR in enumerate(WXR_list):
			_, temp = WXR.split('31/')
			ID, date, time, CR_ = temp.split('_')

			path_to = os.path.join(savepath_WXR, prefix_WXR + y_WXR, ID + '_' + date + time + dcm_ext)
			shutil.copy2(WXR, path_to)

	return True


def DataArrangement_WXRwCT(fd_list = 'Data_List', fd_WXRwCT = '../KJY_PNS_Radiographs_20190814', savepath_WXRwCT = '../PNS_WXRwCT'):

	prefix_CT = 'CT_'
	prefix_WXRwCT = 'WXRwCT_'
	file_ext = '.txt'
	dcm_ext = '.dcm'
	jpg_ext = '.jpg'

	Year_CT = ['2009', '2011', '2012', '2014', '2015', '2016', '2017', '2018', '2019']

	for i, y_CT in enumerate(Year_CT):
		Make_Directory(os.path.join(savepath_WXRwCT, prefix_WXRwCT + y_CT))

		fn = os.path.join(fd_list, prefix_CT + y_CT + file_ext)
		PID = ReadFileLines(fn)
		LPID = RenameID(PID)	

		dataDir = os.path.join(fd_WXRwCT, y_CT)
		all_WXRwCT_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(dataDir) for f in files if all(s in f for s in ['.dcm'])]
		all_WXRwCT_list.sort()

		WXRwCT_ID = []
		WXRwCT_list = []
		for j, WXRwct in enumerate(all_WXRwCT_list):
			_, tmp1 = WXRwct.split('31/')
			tmp2, dcmfile = tmp1.split('/')
			ID, date, time, CR = tmp2.split('_')

			if set([ID])&set(LPID):

				if not set([ID])&set(WXRwCT_ID):
					WXRwCT_ID.append(ID)
					WXRwCT_list.append(WXRwct)
				else:
					__, _tmp1 = WXRwCT_list[-1].split('31/')
					_tmp2, _dcmfile = _tmp1.split('/')
					_ID, _date, _time, _CR = _tmp2.split('_')

					DATE_WXRwCT_list = int(_date + _time)
					date_WXRwct = int(date + time)

					if date_WXRwct < DATE_WXRwCT_list:
						WXRwCT_list[-1] = WXRwct	


		for k, WXRwCT in enumerate(WXRwCT_list):

			_, temp = WXRwCT.split('31/')
			ID, date, time, CR = temp.split('_')

			path_to = os.path.join(savepath_WXRwCT, prefix_WXRwCT + y_CT, ID + '_' + date + time + dcm_ext)
			shutil.copy2(WXRwCT, path_to)

	return True



def DCM_LUT(dcm):

	WC = np.float(dcm.WindowCenter)
	WW = np.float(dcm.WindowWidth)

	img = dcm.pixel_array.astype('float')
	
	p_min = 0.
	p_max = 255.
	rangeFrom = WC - 0.5 - (WW-1)/2
	rangeTo = WC - 0.5 + (WW-1)/2
	

	for idx, p in np.ndenumerate(img):
		if p <= rangeFrom:
			img[idx] = p_min
		elif p > rangeTo:
			img[idx] = p_max
		else:
			img[idx] = ( (img[idx] - (WC-0.5)) / (WW-1) + 0.5 ) * (p_max-p_min) + p_min

	return img




def PI(dcm, img):

	if dcm.PhotometricInterpretation == 'MONOCHROME1':
		img = 255. - img

	return img




def DCM2JPG(dcmpath = "/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2019", jpgpath = "/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXR"):
	jpg_ext = '.jpg'

	all_WXR_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(dcmpath) for f in files if all(s in f for s in ['.dcm'])]
	all_WXR_list.sort()

	for idx, f in enumerate(all_WXR_list):
		dcm = pydicom.dcmread(f)
		print(f.split('/')[-1])
		img = DCM_LUT(dcm)
		img = PI(dcm, img)
		img = np.uint8(img)

		d, p, pd, _, dcm_dir, year, base = f.split("/")
		filename, dcm_ext = os.path.splitext(base)

		Make_Directory(os.path.join(jpgpath, year))
		savepath = os.path.join(jpgpath, year, filename + jpg_ext)

		cv2.imwrite(savepath, img)
		

	return True


def DCM_test(dcmpath):
	
	all_WXR_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(dcmpath) for f in files if all(s in f for s in ['.dcm'])]
	all_WXR_list.sort()
	print(len(all_WXR_list))

	for idx, f in enumerate(all_WXR_list):
		dcm = pydicom.dcmread(f)
		print(f.split('/')[-1])
		img = dcm.pixel_array.astype('float')

	return True



def Anonymize_FN(path = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXR', saveFN = 'PNS_AnonymizedJPG'):

	WXR_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path) for f in files if all(s in f for s in ['.jpg'])]
	WXR_list = Shuffle_List(WXR_list, n=3)

	ID_dict = {}
	for i, l in enumerate(WXR_list):
		newID = RenameID([str(i)])[0] + '_wCT.jpg'
		#ID = l.split('/')[-1][0:8]
		#d = {os.path.basename(l): newID}
		#ID_dict.update(d)

		saveDir = l.replace(l.split('/')[-4], saveFN)
		saveDir = saveDir.replace(os.path.basename(saveDir), newID)
		Make_Directory(os.path.dirname(saveDir))
		
		shutil.copy2(l, saveDir)

		fileDir = saveDir.split(saveDir.split('/')[-2])[0]
		fileDir = os.path.join(fileDir, 'ID_list', saveDir.split('/')[-2] + '.txt')
		Make_Directory(os.path.dirname(fileDir))

		f = open(fileDir, 'a')
		f.write(os.path.basename(l) + ',	' + newID + '\n')
		f.close()
		
	return True
		

#def MatchCTwWXR(CT_path = 








#Anonymize_FN(path = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
	



# DataArrangement_WXRwCT()
# DataArrangement_WXR()
# DCM2JPG()


#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2019', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2018', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2017', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2016', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2015', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2014', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2012', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2011', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXRwCT/WXRwCT_2009', jpgpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_JPG/PNS_WXRwCT')


#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2018')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2017')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2016')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2015')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2014')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2013')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2010')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2011')


#DataArrangement_WXR()
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2009')
#DCM2JPG(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2012')
#DCM_test(dcmpath = '/data/PNS_CLASSIFICATION/PNS_DATA/PNS_WXR/WXR_2012')

	
## devide list (0018, 1000)
# 07.02.190		
# 1861			-
# 09.02.195
# 09.02.158		-
# 1156			-




