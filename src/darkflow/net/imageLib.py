import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
from pylab import *
from PIL import *
#%matplotlib inline
import matplotlib
matplotlib.use("TkAgg")
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
#import numpy as np
import argparse
import glob
#import cv2
from skimage.feature import hog
from skimage import data, exposure

def decompose(image):
	if type(image) is not np.ndarray:
		image = cv2.imread(image)
	height, width, channels = image.shape
	upper = image[0:((int)((height/100)*7)), 0:width]  # AA
	lower = image[((int)((height/100)*7)):height, 0:width]  # AA
	#recomposed = np.concatenate((upper, lower), axis=0)
	return [upper,lower]

def recomposeImage(img1,img2):
	if type(img1) is not np.ndarray:
		img1 = cv2.imread(img1)
	if type(img2) is not np.ndarray:
		img2 = cv2.imread(img2)
	return np.concatenate((img1, img2), axis=0)
