import cv2
import numpy as np 
import math

image = cv2.imread('photo.png')
image = cv2.resize(image, (40, 40)) 
cv2.imshow('image',image)
cv2.waitKey(0)

image = image.astype(dtype=np.int_)

row = len(image)
col = len(image[0])

print(image.shape)

map_ = np.zeros((row,col))
max_map = 0
for i in range(row):
	print(i)
	for j in range(col):

		
		for it in range(row):
			for jt in range(col):
				r = image[i][j][0]
				g = image[i][j][1]
				b = image[i][j][2]
				rt = image[it][jt][0]
				gt = image[it][jt][1]
				bt = image[it][jt][2]

				clr_dist = math.sqrt(pow(abs(r-rt),2) + pow(abs(g-gt),2) + pow(abs(b-bt),2))
				spc_dist = math.sqrt(pow(abs(i-it),2) + pow(abs(j-jt),2))

				map_[i][j] += clr_dist/(math.exp(spc_dist))
				# if(map_[i][j]>max_map):
				# 	max_map = map_[i][j]


# map_ = np.log(1+100*map_)
print(np.amax(map_))
for i in range(row):
	for j in range(col):
		map_[i][j] = (map_[i][j]/np.amax(map_))*255
print(np.amax(map_))

map_ = map_.astype(dtype = np.uint8)

cv2.imwrite('output2.jpg',map_)
# cv2.waitKey(0)




