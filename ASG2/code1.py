import warnings
warnings.filterwarnings("ignore")
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import math
import cv2
import sys


def dist_rgb(a,b):
	dist = pow(a[0]-b[0],2)+pow(a[1]-b[1],2)+pow(a[2]-b[2],2)
	dist = math.sqrt(dist)
	return dist

def ndist(a,b,dig):
	dist = pow(a[0]-b[0],2)+pow(a[1]-b[1],2)
	dist = dist/dig
	dist = 1/np.exp(dist)
	return dist



np.set_printoptions(threshold=sys.maxsize)
image = cv2.imread('Cap2.png')
print(image.shape)

#getting segmets
segments = slic(img_as_float(image), n_segments = 100, sigma = 0, max_iter = 10,compactness = 10.0)
# segments contains the information "segment number for each pixel"

print(segments.shape)
print(np.amax(segments))
num_segment = np.max(segments)


#getting super pixel coler value "average is taken among all pixel in that sp"
cnt = [0.0]*num_segment
spi = []
for i in range(num_segment):
	spi.append([0.0,0.0,0.0])

for i in range(len(segments)):
	for j in range(len(segments[0])):
		ind = segments[i][j]
		# if(ind!=0):
		ind-=1
		cnt[ind]+=1
		spi[ind][0]+=image[i][j][0]
		spi[ind][1]+=image[i][j][1]
		spi[ind][2]+=image[i][j][2]

for i in range(num_segment):
	spi[i][0] /= cnt[i]
	spi[i][1] /= cnt[i]
	spi[i][2] /= cnt[i]

# getting the average coordinate for the super pixel value
regions = regionprops(segments)
print(len(regions))
spc = []
for props in regions:
    cx, cy = props.centroid
    spc.append([cx,cy])

print(len(spi))
print(len(spc))
super_pixel_values = np.float32(np.array(spi))

k = 6
kmeans = KMeans(n_clusters=6, random_state=0).fit(super_pixel_values)
labels = kmeans.labels_
# print(labels.shape)
# exit(0)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
# retval, labels, centers = cv2.kmeans(super_pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
# print(labels)
# print(labels.shape)

m,n = image.shape[:2]
N = len(labels)
print("N", N)
cluster_sp_count = np.zeros(6)
for i in labels:
	cluster_sp_count[i]+=1

print(cluster_sp_count)
# exit(0)
image_labels = np.zeros((m,n))

for i in range(len(segments)):
	for j in range(len(segments[0])):
		seg_num = segments[i][j]
		image_labels[i][j] = labels[seg_num-1]

# print(image_labels)
print(image_labels.shape)

np_image = np.float32(image)
cluster = []
cluster_sp_sp = []
for i in range(k):
	cluster.append([])
	cluster_sp_sp.append([])

for i in range(len(labels)):
	# print(labels[i])
	# print
	cluster_sp_sp[labels[i]].append(spc[i])



for i in range(m):
	for j in range(n):
		cluster[int(image_labels[i][j])].append([i,j])

print("length of each clusters", end = " ")
for i in range(k):
	print(len(cluster[i]),end = " ")
print()

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


#contrast cue
contrast = [0]*k
for K in range(k):
	for i in range(k):
		if(K!=i):
			contrast[K]+=(cluster_sp_count[i]/N)*dist_rgb(color_avg[K],color_avg[i])
			# contrast[K] += (len(cluster[i])/N)*dist_rgb(color_avg[K],color_avg[i])


#spatial cue
spatial = [0]*k
dig = math.sqrt(pow(m,2)+pow(n,2))

for K in range(k):
	temp = 0
	for sp_dist in cluster_sp_sp[K]:
		temp += ndist(sp_dist,spatial_centers[K],dig)
	# temp = temp/len(cluster[K])
	temp = temp/cluster_sp_count[K]
	spatial[K] = temp

contrast = np.float32(np.array(contrast))
spatial = np.float32(np.array(spatial))

contrast_scale = (contrast/np.max(contrast))*255
spatial_scale = (spatial/np.max(spatial))*255

print(contrast_scale)
print(spatial_scale)

contrast_cue = np.zeros((m,n))
spatial_cue = np.zeros((m,n))

for K in range(k):
	for pts in cluster[K]:
		contrast_cue[pts[0]][pts[1]] = contrast_scale[K]

for K in range(k):
	for pts in cluster[K]:
		spatial_cue[pts[0]][pts[1]] = spatial_scale[K]

contrast_image = contrast_cue
# contrast_image = np.uint8(contrast_cue)
plt.imshow(contrast_image,cmap="gray")
plt.show()

cv2.imwrite("contrast_cue.png", contrast_image)

spatial_image = spatial_cue
# spatial_image = np.uint8(spatial_cue)
plt.imshow(spatial_image,cmap="gray")
plt.show()
cv2.imwrite("spatial_cue.png", spatial_image)

# print(contrast)
# print(spatial)

#plotting the super pixels
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()

