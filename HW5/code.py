import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('straw.png')
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
			if(temp>curr):
				strr = strr+ '1'
			else:
				strr = strr+'0'
		lbp[i-1][j-1] = int(strr,2)

part = 2
prevr = 0
curr = 300
patches = []
for r in range(part):
	prevc = 0
	curc = 200
	for c in range(part):
		patch = lbp[prevr:curr, prevc:curc]
		temp = [0]*256
		for i in range(len(patch)):
			for j in range(len(patch[0])):
				temp[int(patch[i][j])]+=1
		prevc = curc
		curc = curc+curc
		patches.extend(temp)
		plt.plot(temp)
		plt.show()
	prevr = curr
	curr = curr+curr
print((patches))
plt.show()

cv2.imwrite('output.png',lbp)




cv2.imshow('image',lbp)
cv2.waitKey(0)

