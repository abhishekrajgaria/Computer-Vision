import warnings
warnings.filterwarnings("ignore")
import math
import cv2
import sys
import numpy as np
from fcmeans import FCM
import matplotlib.pyplot as plt 


def dist_rgb(a,b):
	dist = pow(a[0]-b[0],2)+pow(a[1]-b[1],2)+pow(a[2]-b[2],2)
	dist = math.sqrt(dist)
	return dist

image = cv2.imread('cap2.png')
# image = cv2.resize(image,(300,300))
print(image.shape)
m,n = image.shape[:2]
rgbxy = np.zeros((m,n,5))

for i in range(m):
	for j in range(n):
		rgbxy[i][j][0] = image[i][j][0]/255
		rgbxy[i][j][1] = image[i][j][1]/255
		rgbxy[i][j][2] = image[i][j][2]/255
		rgbxy[i][j][3] = i/m
		rgbxy[i][j][4] = j/n

print(rgbxy.shape)
values = np.float32(rgbxy.reshape((-1,5)))
print(values[:5])
# norm_values = values / values.max(axis = 0)
norm_values = values
print(norm_values[:5])
print(values.shape)
print(norm_values.shape)
norm_values = np.float32(norm_values)

fcm = FCM(n_clusters = 12)
fcm.fit(norm_values)
fcm_centers = fcm.centers
fcm_labels = fcm.u.argmax(axis=1)
print(fcm_centers.shape)
print(fcm_labels.shape)

dict_count = {}
for i in fcm_labels:
	if(i not in dict_count):
		dict_count[i] = 0
	dict_count[i]+=1
print(dict_count)

segmented_image = fcm_centers[fcm_labels.flatten()][:,:3]
print(segmented_image.shape)
segmented_image = segmented_image.reshape(image.shape)
# segmented_image = (segmented_image/np.amax(segmented_image))*255.0
segmented_image = segmented_image*255
segmented_image = np.uint8(segmented_image)
cv2.imshow("segmented_image",segmented_image)
cv2.waitKey(0)
cv2.imwrite("segmented_image.png", segmented_image)
kernel = np.ones((7,7), np.uint8)

dilated_image = cv2.dilate(segmented_image, kernel, iterations=1)
eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

dict_color = {}
for i in range(m):
	for j in range(n):
		color = (segmented_image[i][j][0],segmented_image[i][j][1],segmented_image[i][j][2])
		dict_color[color] = 1
print(len(dict_color))
print(dict_color)


# cv2.imshow("dilated_image",dilated_image)
# cv2.waitKey(0)
# print(eroded_image)
eroded_image = segmented_image

def update_image(image):
	m1,n1 = image.shape[:2]
	for i in range(m1):
		for j in range(n1):
			temp = []
			current_tuple = (image[i][j][0],image[i][j][1],image[i][j][2])
			minn = 100000000
			ch_ind = 0
			itr = 0
			keys = list(dict_color.keys())#.tolist()
			for key in keys:
				dist = dist_rgb(key,current_tuple)
				if(dist<minn):
					minn = dist
					ch_ind = itr
				itr+=1
			image[i][j][0] = keys[ch_ind][0]
			image[i][j][1] = keys[ch_ind][1]
			image[i][j][2] = keys[ch_ind][2]

	return image
eroded_image = update_image(eroded_image)

dict_color = {}
for i in range(m):
	for j in range(n):
		color = (eroded_image[i][j][0],eroded_image[i][j][1],eroded_image[i][j][2])
		dict_color[color] = 1
print(len(dict_color))
print(dict_color)

# cv2.imshow("eroded_image",eroded_image)
# cv2.waitKey(0)



# print(fcm_centers)
cluster_center = fcm_centers[:,:3]
# cluster_center = np.uint8((cluster_center/np.amax(cluster_center))*255.0) #column wise
cluster_center = np.uint8(cluster_center*255)
print(cluster_center)
print(cluster_center.shape)

image_labels = np.zeros((m,n))
for i in range(m):
	for j in range(n):
		for l in range(len(cluster_center)):
			if(eroded_image[i][j][0]==cluster_center[l][0] and eroded_image[i][j][1]==cluster_center[l][1] and eroded_image[i][j][2]==cluster_center[l][2]):
				image_labels[i][j] = l 
				break
image_labels = np.uint8(image_labels)

def merge(image,image_labels,cluster_center,stride=1 ,size = 3):
	m1,n1 = image_labels.shape
	for i in range(0,m1-size+1,stride):
		for j in range(0,n1-size+1,stride):
			# sub_image = image_labels[i:i+size,j:j+size]
			dict_label = {}
			maxx = 0
			fin_label = 0
			for i1 in range(size):
				for j1 in range(size):
					curr_label = image_labels[i+i1][j+j1]
					if(curr_label not in dict_label):
						dict_label[curr_label] = 0
					dict_label[curr_label]+=1
					if(dict_label[curr_label]>maxx):
						maxx = dict_label[curr_label]
						fin_label = curr_label
			for i1 in range(size):
				for j1 in range(size):
					image_labels[i+i1][j+j1] = int(fin_label)

	for i in range(m1):
		for j in range(n1):
			# print(image[i][j][0])
			# print(image_labels[i][j])
			# print(cluster_center[image_labels[i][j]][0])
			image[i][j][0] = cluster_center[image_labels[i][j]][0]
			image[i][j][1] = cluster_center[image_labels[i][j]][1]
			image[i][j][2] = cluster_center[image_labels[i][j]][2] 
	return image

eroded_image = merge(eroded_image,image_labels,cluster_center)
cv2.imshow("eroded_image",eroded_image)
cv2.waitKey(0)

# num_labels, labels_im = cv2.connectedComponents(eroded_image)
# print(num_labels)
# print(labels_im)

print()
