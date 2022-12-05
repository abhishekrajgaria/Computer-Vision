import numpy as np 
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('dog3.png')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(image.shape)

fg = cv2.imread('fg3.png')
fg = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)
print(fg.shape)
bg = cv2.imread('bg3.png')
bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
print(bg.shape)

fg_hist = [0]*256
bg_hist = [0]*256

fg_map = [0]*256
bg_map = [0]*256

for i in range(len(fg)):
	for j in range(len(fg[0])):
		fg_hist[fg[i][j]] += 1


for i in range(len(bg)):
	for j in range(len(bg[0])):
		bg_hist[bg[i][j]] += 1

# print(max(fg_hist))
for i in range(256):
	fg_hist[i] /= max(fg_hist) 
# for i in fg_hist:
# 	i = i/max(fg_hist)
for i in range(256):
	bg_hist[i] /= max(bg_hist) 


# plt.plot(fg_hist)
# # plt.show()

# plt.plot(bg_hist)
# plt.show()
for i in range(256):
	fg_map[i] = fg_hist[i]*256

for i in range(256):
	bg_map[i] = bg_hist[i]*256

fg_image = np.zeros(image.shape)

for i in range(len(image)):
	for j in range(len(image[0])):
		fg_image[i][j] = fg_map[image[i][j]]


bg_image = np.zeros(image.shape)

for i in range(len(image)):
	for j in range(len(image[0])):
		bg_image[i][j] = bg_map[image[i][j]]


sg_hist = [0]*256
for i in range(256):
	sg_hist[i] = (fg_hist[i] + (1 - bg_hist[i]))/2  

plt.plot(sg_hist)
plt.show()

sg_map = [0]*256
for i in range(256):
	sg_map[i] = sg_hist[i]*256

sg_image = np.zeros(image.shape)

for i in range(len(image)):
	for j in range(len(image[0])):
		sg_image[i][j] = sg_map[image[i][j]]

cv2.imshow('fg_image',fg_image)
cv2.waitKey(0)

cv2.imshow('bg_image',bg_image)
cv2.waitKey(0)

# sg_image = sg_image.astype(dtype = np.uint8)
cv2.imwrite('output3.png',sg_image)
