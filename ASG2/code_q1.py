import warnings
warnings.filterwarnings("ignore")
import math
import cv2
import sys
import numpy as np
from fcmeans import FCM
import matplotlib.pyplot as plt 
from skimage import morphology
from scipy.ndimage import distance_transform_edt


image = cv2.imread('cap2.png')
# image = cv2.resize(image,(300,300))
print(image.shape)
m,n = image.shape[:2]
rgbxy = np.zeros((m,n,5))

for i in range(m):
	for j in range(n):
		rgbxy[i][j][0] = image[i][j][0]
		rgbxy[i][j][1] = image[i][j][1]
		rgbxy[i][j][2] = image[i][j][2]
		rgbxy[i][j][3] = i
		rgbxy[i][j][4] = j

print(rgbxy.shape)
values = np.float32(rgbxy.reshape((-1,5)))
print(values[:5])
norm_values = values / values.max(axis = 0)
# norm_values = values
print(norm_values[:5])
print(values.shape)
print(norm_values.shape)
norm_values = np.float32(norm_values)
k = 8
fcm = FCM(n_clusters = k)
fcm.fit(norm_values)
centers = fcm.centers
labels = fcm.u.argmax(axis=1)
print(centers.shape)
print(labels.shape)

labels = (labels.reshape((m,n)))
segmented_image = np.zeros((m,n,3))
for i in range(m):
	for j in range(n):
		labels[i][j] = int(labels[i][j])
		segmented_image[i,j,:] = centers[labels[i][j]][:3]

cv2.imshow("segmented_image",segmented_image)
cv2.waitKey(0)
cv2.imwrite("segmented_image.png",segmented_image*255)

image_holes = np.zeros(labels.shape)
boolean_image = np.zeros(labels.shape)
for K in range(k):
	boolean_image += morphology.remove_small_objects(labels==K,100)
	image_holes += np.uint8(morphology.remove_small_objects(labels==K, 100))*255
cv2.imshow("image",image_holes)
cv2.waitKey(0)
cv2.imwrite("image_holes.png",image_holes)

new_index_hole = tuple(distance_transform_edt(np.logical_not(boolean_image),
 return_indices=True, return_distances=False))
print(type(new_index_hole))

new_merged_image = segmented_image[new_index_hole]
cv2.imshow("new_merged_image",new_merged_image)
cv2.waitKey(0)
cv2.imwrite("new_merged_image.png", new_merged_image*255)