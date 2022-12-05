import numpy as np
import math
from sklearn.cluster import KMeans
import cv2


def dist(a,b):
	return abs(a-b)
	# return math.sqrt(pow(a-b,2))

image = np.array([[10,5,1],[4,10,2],[11,1,12]])
mod_img = image.reshape((-1))
print(image.shape)
# image = np.array(image)
mod_img = np.float32(mod_img)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print(mod_img.shape)

flatten_image = image.flatten()
data_points = []
for i in range(len(flatten_image)):
	data_points.append([flatten_image[i]])
data_points = np.array(data_points)
unique = np.unique(data_points)
# print(unique)
# # data_points.reshape(9,1)
# print(data_points)
# print(data_points.shape)
optimal_clus_num = 0
optimal_sl_coef = 0
for n_clus in range(2,len(flatten_image)):
	comp,labels,centers = cv2.kmeans(mod_img,n_clus,None,criteria,10,cv2.KMEANS_PP_CENTERS)
	# print(comp)
	# print(labels)
	labels_ = labels.reshape(9)
	print(labels)
	# print(centers)
	# break
# 	kmeans = KMeans(n_clusters=n_clus, random_state=0).fit(data_points)
	# labels_ =kmeans.labels_
# 	print(labels_)
	cluster_count = [0]*(len(np.unique(labels_)))
	for i in labels_:
		cluster_count[i]+=1
	# print(cluster_count)
	cluster_points = []
	for i in range(len(cluster_count)):
		temp = []
		for j in range(len(data_points)):
			if(labels_[j]==i):
				temp.append(data_points[j][0])
		cluster_points.append(temp)
	# print(cluster_points)
	a = [0]*len(data_points)
	b = [0]*len(data_points)
	s = [0]*len(data_points)
	for i in range(len(data_points)):
		clus = labels_[i]
		if(cluster_count[clus]>1):
			curr_pt = data_points[i][0]
			for pt in cluster_points[clus]:
				a[i] += dist(curr_pt,pt)
			a[i] = a[i]/(cluster_count[clus]-1)
			min_mean = math.inf
			for s_clus in range(len(cluster_count)):
				if(s_clus!=clus):
					temp_mean = 0
					for pt in cluster_points[s_clus]:
						temp_mean+=dist(curr_pt,pt)
					temp_mean = temp_mean/cluster_count[s_clus]
					min_mean = min(min_mean,temp_mean)
					b[i] = min_mean
			s[i] = (b[i]-a[i])/max(b[i],a[i])
		else:
			s[i] = 0

	s = np.array(s)
	mean_slh = np.mean(s)
	print(s)
	print(mean_slh)
	print(n_clus)
	print("++++++++++++++++++++++++")
	if(mean_slh > optimal_sl_coef):
		optimal_sl_coef = mean_slh
		optimal_clus_num = n_clus

print("===================================")
print(optimal_clus_num)
print(optimal_sl_coef)

