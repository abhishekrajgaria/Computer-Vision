import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 
import math


def dist_rgb(a,b):
	dist = pow(a[0]-b[0],2)+pow(a[1]-b[1],2)+pow(a[2]-b[2],2)
	dist = math.sqrt(dist)
	return dist

def ndist(a,b,dig):
	dist = pow(a[0]-b[0],2)+pow(a[1]-b[1],2)
	dist = dist/dig
	dist = 1/math.exp(dist)
	return dist


image = cv.imread('Cap2.png')

print(image.shape)
pixel_values = image.reshape((-1,3))
print(image.shape)
pixel_values = np.float32(pixel_values)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
k = 6
retval, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS) 
m = len(image)
n = len(image[0])
labels=labels.reshape((m,n))


cluster = []
for i in range(k):
	cluster.append([])

for i in range(len(image)):
	for j in range(len(image[0])):
		cluster[labels[i][j]].append([i,j])

for i in range(k):
	print(len(cluster[i]))
color_avg = []
for i in range(k):
	temp = [0,0,0]
	for pts in cluster[i]:
		temp[0]+=image[pts[0]][pts[1]][0]
		temp[1]+=image[pts[0]][pts[1]][1]
		temp[2]+=image[pts[0]][pts[1]][2]
	temp[0] /= len(cluster[i])
	temp[1] /= len(cluster[i])
	temp[2] /= len(cluster[i])
	color_avg.append(temp)


spatial_centers = []
for i in range(k):
	temp = [0,0]
	for pts in cluster[i]:
		temp[0] += pts[0]
		temp[1] += pts[1]
	temp[0] /= len(cluster[i])
	temp[1] /= len(cluster[i])
	spatial_centers.append(temp)


image_center = [0,0]
for i in range(m):
	for j in range(n):
		image_center[0]+=i
		image_center[1]+=j
image_center[0]/=(m*n)
image_center[1]/=(m*n)

dig = math.sqrt(pow(len(image),2)+pow(len(image[0]),2))

# computing Contrast cue 
contrast = [0]*k
N = m*n
for K in range(k):
	for i in range(k):
		if(K!=i):
			contrast[K] += (len(cluster[i])/N)*dist_rgb(centers[K],centers[i])

contrast = np.array(contrast)

# print(contrast)


#computing spatial cue

spatial = [0]*k

for K in range(k):
	temp = 0
	for pts in cluster[K]:
		temp += ndist(pts,spatial_centers[K],dig)
	temp = temp/len(cluster[K])
	spatial[K] = temp


print(centers)
print(" checker ====================")
print(color_avg)

exit(0)

spatial = np.array(spatial)
print(spatial)


#computing saliency cue

saliency = [0]*k
for i in range(k):
	saliency[i] = contrast[i]*spatial[i]


maxi = np.max(saliency)

for i in range(k):
	saliency[i] = (saliency[i]/maxi)*255

print(saliency)

out=np.zeros(labels.shape)

for i in range(m):
	for j in range(n):
		out[i][j]=saliency[labels[i][j]]

out = np.uint8(out)
plt.imshow(out,cmap="gray")
plt.show()
