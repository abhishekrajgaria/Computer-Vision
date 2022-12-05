import warnings
warnings.filterwarnings("ignore")
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)
image = cv2.imread('eagle.jpg')
print(image.shape)
segments = slic(img_as_float(image), n_segments = 100, sigma = 0, max_iter = 10,compactness = 10.0)
print(np.amax(segments))
num_segment = np.max(segments)
# print(segments)
print(segments.shape)
print(segments[0].shape)
print(segments[0][0], segments[1][0])

# num_segment = np.unique(segments)
# super_pixel_list = [np.where(segments==i) for i in num_segment]
# print(super_pixel_list[0])

num_segment
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


regions = regionprops(segments)
print(len(regions))
spc = []
for props in regions:
    cx, cy = props.centroid
    spc.append([cx,cy])


print(spi[0])
print("hello")
spo = [0]*num_segment

for i in range(num_segment):
	temp = 0
	for j in range(num_segment):
		clr_dist = math.sqrt(pow((spi[i][0] - spi[j][0]),2) + pow((spi[i][1] - spi[j][1]),2) + pow((spi[i][2] - spi[j][2]),2))
		spc_dist = math.sqrt(pow((spc[i][0] - spc[j][0]),2) + pow((spc[i][1] - spc[j][1]),2))
		spo[i] += (clr_dist/math.exp(spc_dist/(math.sqrt(2)*256)))
out_image = np.zeros(segments.shape)

for i in range(len(segments)):
	for j in range(len(segments[0])):
		ind = segments[i][j]
		# if(ind!=0):
		ind-=1
		out_image[i][j] = spo[ind]

map_ = out_image
print(np.amax(map_))
val = np.max(map_)
for i in range(len(map_)):
	for j in range(len(map_[0])):
		map_[i][j] = (map_[i][j]/val)*255
print(np.amax(map_))

map_ = map_.astype(dtype = np.uint8)


cv2.imwrite('output2.jpg',map_)
cv2.imshow('output',map_)
cv2.waitKey(0)

# print(spc)







fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()