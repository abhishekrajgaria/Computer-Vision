import cv2
import math
import numpy as np 

image = cv2.imread('bird.jpg')

image = cv2.resize(image, (240, 240)) 
print(image.shape)
cv2.imshow('image',image)
cv2.waitKey(0)

spp = []

dividor = [4,9,16,25]
# dividor.reverse()

for d in dividor:
	part = int(math.sqrt(d))
	inc = 240//part
	prevr = 0
	curr = inc
	

	for i in range(part):
		prevc = 0
		curc = inc
		for j in range(part):
			# value = 0
			# for it in range(120):
			# 	for jt in range(120):
			# 		value += image[it][jt][0]
			# print(value/14400)
			sub_matrix = image[prevr:curr,prevc:curc]
			# print(type(sub_matrix))
			# print(sub_matrix)
			# print(sub_matrix[:,:,0].shape)

			matr = (sub_matrix[:,:,0].flatten())
			matg = (sub_matrix[:,:,1].flatten())
			matb = (sub_matrix[:,:,2].flatten())



			# print(matr)
			
			mean1 = np.mean(matr)
			mean2 = np.mean(matg)
			mean3 = np.mean(matb)

			std1 = np.std(matr)
			std2 = np.std(matg)
			std3 = np.std(matb)

			spp.extend([mean1,mean2,mean3,std1,std2,std3])
			print(len(spp))
			# exit(0)
			prevc = curc
			curc = curc+inc
		prevr = curr
		curr += inc
# spp.sort()
print(spp)



