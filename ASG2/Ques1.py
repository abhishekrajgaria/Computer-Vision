import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import os
import cv2
import numpy as np
from time import time

img = cv2.imread("Cap2.PNG")

print(img.shape)
img = cv2.resize(img, (243,326), interpolation=cv2.INTER_AREA)  
print(img.shape)

shape = np.shape(img)
num_clusters = 10
rgb_xy = np.zeros((200*200,5))
counter = 0
for i in range(200):
    for j in range(200):
        rgb_xy[counter,0] = img[i,j,2]
        rgb_xy[counter,1] = img[i,j,1]
        rgb_xy[counter,2] = img[i,j,0]
        rgb_xy[counter,3] = i
        rgb_xy[counter,4] = j
        counter+=1

rgb_xy = rgb_xy/np.max(rgb_xy,axis=0)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
rgb_xy.T, num_clusters, 2, error=0.001, maxiter=1000, seed=42)
out_img = np.zeros((200,200))
out_img_col = np.zeros((200,200,3))
counter=0
for x in range(200):
    for y in range(200):
        ind = np.argmax(u.T[counter])
        out_img[x,y] = int(ind)
        out_img_col[x,y,:] = cntr[int(ind),0:3]
        counter+=1
print("1")

from skimage import morphology
mask = np.zeros((out_img.shape[0],out_img.shape[1]))
for ind in range(num_clusters):
    mask += morphology.remove_small_objects(out_img==ind)
print("1")
from scipy.ndimage import distance_transform_edt as dt
ind = tuple(dt(np.logical_not(mask), return_indices = True, return_distances = False))

print(ind.shape)
new_img_col = out_img_col[ind]
print("1")
plt.figure(figsize=(20,20))
plt.subplot(1,4,1)
plt.imshow(img)
plt.title("Input Image")

plt.subplot(1,4,2)
plt.imshow(out_img_col)
plt.title("Clustered Image")

plt.subplot(1,4,3)
plt.imshow(new_img_col)
plt.title("Clustered Image after Merging Isolated Pixels")
plt.show()