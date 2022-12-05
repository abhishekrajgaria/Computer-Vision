import cv2
import numpy as np

image = cv2.imread('iiitd1.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
row = len(gray)
col = len(gray[0])
min_o = -1
thresh = -1
for t in range(0,256,2):
	pl = []
	ph = []
	for i in range(row):
		for j in range(col):
			if(gray[i][j]<=t):
				pl.append(gray[i][j])
			elif(gray[i][j]>t):
				ph.append(gray[i][j])

	pl = np.array(pl)
	ph = np.array(ph)
	if(len(pl)==0):
		m0 = 0
	else:
		m0 = np.mean((pl))
	
	if(len(ph)==0):
		m1 = 0
	else:
		m1 = np.mean((ph))

	tssl = np.sum(np.square(pl - m0))
	tssh = np.sum(np.square(ph - m1))

	o = tssl + tssh
	if(min_o<0):
		min_o = o
		thresh = t
	else:
		if(min_o > o ):
			min_o = o
			thresh = t
	print(thresh)
print("threshold obtained!")

image1 = np.zeros(gray.shape)
image2 = np.zeros(gray.shape)


cnt1 = 0
cnt2 = 0
for i in range(row):
	for j in range(col):
		if(gray[i][j]<thresh):
			image1[i][j] = 1
			cnt1+=1
		else:
			cnt2+=1
			image2[i][j] = 1

print(cnt1,cnt2)
# cv2.imshow('image', image1)
# cv2.waitKey(0)
# cv2.imshow('image', image2)
# cv2.waitKey(0)

fg = 0
bg = 0

#selection using count
if(cnt1<cnt2):
	fg = image2
	bg = image1
	cv2.imshow('image',image1)
	cv2.waitKey(0)
else:
	fg = image1
	bg = image2
	cv2.imshow('image',image2)
	cv2.waitKey(0)

cv2.imshow('output_image',fg)
cv2.waitKey(0)

