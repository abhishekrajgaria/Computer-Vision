import cv2
import numpy as np
from scipy import signal


image = cv2.imread('bike.jpg')
print(image.shape)
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
piwrett_1= [[1,0,-1],[1,0,-1],[1,0,-1]]
pa = np.zeros((3,3))
for i in range(3):
	for j in range(3):
		pa[i][j] = piwrett_1[i][j]

pb = np.transpose(pa)
# ex = signal.convolve2d(image,pa,mode='same')
# ey = signal.convolve2d(image,pb,mode='same')

# output = np.sqrt(np.square(ex)+np.square(ey))

# output = output.astype('uint8')

# cv2.imwrite('output.png',output)

n,m = image.shape[0:2]

# image = image.astype('float64')

edgex = np.zeros((n-2,m-2))
edgey = np.zeros((n-2,m-2))

for i in range(n-2):
	for j in range(m-2):
		temp = 0
		for k in range(3):
			for l in range(3):
				temp += image[i+k][j+l]*pa[k][l]
		edgex[i][j] = temp

for i in range(n-2):
	for j in range(m-2):
		temp = 0
		for k in range(3):
			for l in range(3):
				temp += image[i+k][j+l]*pb[k][l]
		edgey[i][j] = temp

output = np.sqrt(np.square(edgex)+np.square(edgey))

# output = output.astype('uint8')

cv2.imwrite('output.png',output)

# edgex = edgex.astype('uint8')
# edgey = edgey.astype('uint8')

# print()

# cv2.imshow('image',edgex)
# cv2.waitKey(0)


# cv2.imshow('image',edgey)
# cv2.waitKey(0)

