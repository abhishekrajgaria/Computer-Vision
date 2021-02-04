import cv2
import numpy as np

image = cv2.imread('dogimage.jpg')

# image.show()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
row = len(gray)
col = len(gray[0])

min_o = -1
thresh = -1
for t in range(0,256):
	w0 = 0
	w1 = 0
	pl = []

	ph = []
	for i in range(row):
		for j in range(col):
			if(gray[i][j]<t):
				w0+=1
				pl.append(gray[i][j])
			elif(gray[i][j]>=t):
				w1+=1
				ph.append(gray[i][j])

	if(len(pl)==0):
		v0 = 0
	else:
		v0 = np.var(np.array(pl))
	
	if(len(ph)==0):
		v1 = 0
	else:
		v1 = np.var(np.array(ph))

	o =w0*v0 + w1*v1
	if(min_o<0):
		min_o = o
		thresh = t
	else:
		if(min_o > o ):
			min_o = o
			thresh = t
	print(thresh)
print(thresh)

# 92, 93, 95 could be used as threshold value obtained from the above code
# with changing the equality of t

image1 = np.zeros(gray.shape)
image2 = np.zeros(gray.shape)

thresh = 93

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
cv2.imshow('image', image1)
cv2.waitKey(0)
cv2.imshow('image', image2)
cv2.waitKey(0)


cnt3 = 0
cnt4 = 0

cnt_row = row//2-100
cnt_col = col//2-100
for i in range(cnt_row,cnt_row+200):
	for j in range(cnt_col,cnt_col+200):
		if(image1[i][j]==1):
			cnt3+=1
		if(image2[i][j]==1):
			cnt4+=1
print(cnt3,cnt4)

cnt5 = 0
cnt6 = 0
border = []
for i in range(row):
	for j in range(col):
		if(i<50 or j<50 or i>684 or j>909):
			border.append([i,j])

for x in border:
	i = x[0]
	j = x[1]
	if(image1[i][j]==1):
		cnt5+=1
	if(image2[i][j]==1):
		cnt6+=1

print(cnt5, cnt6)
fg = 0
bg = 0

#selection using count
if(cnt1<cnt2):
	fg = image1
	bg = image2
	cv2.imshow('image',image1)
	cv2.waitKey(0)
else:
	fg = image2
	bg = image1
	cv2.imshow('image',image2)
	cv2.waitKey(0)

#selection using center count
if(cnt3>cnt4):
	fg = image1
	bg = image2
	cv2.imshow('image',fg)
	cv2.waitKey(0)
else:
	fg = image2
	bg = image1
	cv2.imshow('image',fg)
	cv2.waitKey(0)


#selection using border count
if(cnt5<cnt6):
	fg = image1
	bg = image2
	cv2.imshow('image',fg)
	cv2.waitKey(0)
else:
	fg = image2
	bg = image1
	cv2.imshow('image',fg)
	cv2.waitKey(0)

ffg = image.copy()
fbg = image.copy()

for i in range(row):
	for j in range(col):
		if(fg[i][j]!=1):
			ffg[i][j][0] = 200
			ffg[i][j][1] = 0
			ffg[i][j][2] = 0

cv2.imshow('image',ffg)
cv2.waitKey(0)

for i in range(row):
	for j in range(col):
		if(bg[i][j]!=1):
			fbg[i][j][0] = 200
			fbg[i][j][1] = 0
			fbg[i][j][2] = 0

cv2.imshow('image',fbg)
cv2.waitKey(0)

# bgr = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
# rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
# rgb[fg!=1] = (100,0,0)
# cv2.imshow('image',rgb)
# cv2.waitKey(0)


# 	o_values.append(w0*v0 + w1*v1)

# o_values.sort()
# print(o_values[0])
