import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('iiitd1.png')
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
lbp = np.zeros(image.shape)
image = cv2.copyMakeBorder(image, 1,1,1,1,cv2.BORDER_CONSTANT)
xmoves = [-1,-1,-1, 0, 1, 1, 1, 0]
ymoves = [-1, 0, 1, 1, 1, 0,-1,-1]

for i in range(1,len(image)-1):
	for j in range(1,len(image[0])-1):
		strr = ""
		curr = image[i][j]
		for m in range(8):
			x = i + xmoves[m]
			y = j + ymoves[m]
			temp = image[x][y]
			ratio = min(curr,temp)/(max(curr,temp) + 1e-5) 
			if(ratio>=0.5):
				strr = strr+ '1'
			else:
				strr = strr+'0'
		lbp[i-1][j-1] = int(strr,2)


lbp = lbp.astype(dtype = np.uint8)
cv2.imshow('lbp_image',lbp)
cv2.waitKey(0)