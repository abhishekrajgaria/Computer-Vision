import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# limit = 2000000
# import sys
# sys.setrecursionlimit(limit)


class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

class Circle:
	def __init__(self,c,r):
		self.c = c
		self.r = r

def dist(a,b):
	return math.sqrt((a.x -b.x)**2 + (a.y-b.y)**2)

def inside(c,p):
	if(dist(c.c,p)<=c.r):
		return True
	return False

def get_center(v1,v2,v3,v4):
	x = v1*v1 + v2*v2
	y = v3*v3 + v4*v4
	z = v1*v4 - v2*v3
	return Point((v4*x - v2*y)/(2*z) , (v1*y - v3*x)/(2*z))

def gen_circle3(a,b,c):
	cen = get_center(b.x - a.x, b.y - a.y, c.x-a.x, c.y-a.y)
	cen.x+=a.x
	cen.y+=a.y

	return Circle(cen,dist(cen,a))

def gen_circle2(a,b):
	cen = Point((a.x+b.x)/2, (a.y+b.y)/2)
	return Circle(cen, dist(a,b)/2)

def is_circle(c,ps):
	for p in ps:
		if(not inside(c,p)):
			return False
	return True

def create_points(arr):
	ps = []
	for i in arr:
		ps.append(Point(i[0],i[1]))
	return ps

def isvalid(img, i, j, visited):
	row = len(img)
	col = len(img[0])
	return (((i>=0) and i<row and j>=0 and j<col and (img[i][j]>0 and visited[i][j]==0)))
	# return False

def dfs(img, i, j, visited,point_set,count):
	rowmov = [-1,-1,-1,0,0,1,1,1]
	colmov = [-1,0,1, -1, 1, -1, 0, 1]
	stack = []
	stack.append([i,j])
	while(stack != []):
		x = stack.pop()
		r = x[0]
		c = x[1]
		visited[r][c] = 1
		# img[r][c] = (count+1)*25
		point_set.append([r,c])
		for k in range(8):
			if(isvalid(img, r+rowmov[k], c+colmov[k],visited)):
				stack.append([r+rowmov[k],c+colmov[k]])

def count_image(imgmat):

	row = len(imgmat)
	col = len(imgmat[0])
	visited = np.zeros(imgmat.shape)

	count = 0
	array = []
	for i in range(row):
		for j in range(col):
			if(imgmat[i][j]>0 and visited[i][j]==0):
				point_set = []
				dfs(imgmat,i,j,visited,point_set,count)
				count+=1
				array.append(point_set)

	return count,array


def hull_points(arr,hull):
	ps = []
	for i in hull.vertices:
		ps.append(arr[i])
	return ps

def brute_circ(pts):
	minr = math.inf
	ans = 0
	points = create_points(pts)
	npt = len(points)
	for i in range(npt-2):
		for j in range(i+1,npt-1):
			for k in range(j+1,npt):
				c = gen_circle3(points[i],points[j],points[k])
				fg = 1
				for l in points:
					if(not inside(c,l)):
						fg = 0
				if(fg==1 and c.r<minr):
					ans = c
					minr = c.r
	return ans

def jaccard(cir, pts, image):
	bmask1 = np.zeros(image.shape)
	bmask2 = np.zeros(image.shape)
	# print(len(pts))
	for pt in pts:
		bmask1[pt[0]][pt[1]] = 255

	total_pts = len(image)*len(image[0])
	nptcirc = 0
	for i in range(len(image)):
		for j in range(len(image[0])):
			if(inside(cir,Point(i,j))):
				bmask2[i][j] = 255
				nptcirc+=1
	cnt1 = 0
	for i in range(len(bmask1)):
		for j in range(len(bmask1[0])):
			if(bmask1[i][j]>0):
				cnt1+=1
	# print(cnt1,"----")
	cnt2 = 0
	for i in range(len(bmask2)):
		for j in range(len(bmask2[0])):
			if(bmask2[i][j]>0):
				cnt2+=1
	# print(cnt2,"----")
	# print(bmask1.shape)
	# print(bmask2.shape)
	cv2.imshow('bmask1',bmask1)
	cv2.waitKey(0)
	cv2.imshow('bmask2',bmask2)
	cv2.waitKey(0)
	print(cnt1, cnt2, total_pts)
	#considering only the circular region
	j_score = cnt1/cnt2
	print(j_score)

	#considering complete mask region
	# j_score = (cnt1+total_pts-cnt2)/total_pts
	# print(j_score)
	print("--------------")
	return j_score


if __name__ == '__main__':


	image = cv2.imread('binary_image.png')
	image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	print(image.shape)
	nimage, object_points = count_image(image)
	print(nimage)
	centers = []
	for itr in range(nimage):
		hull = ConvexHull(object_points[itr])
		points = hull_points(object_points[itr],hull)
		c = brute_circ(points)
		centers.append(c)
		print(c.c.x,c.c.y,c.r)
		cv2.circle(image, (math.floor(c.c.y),math.floor(c.c.x)), math.floor(c.r), 255, thickness=1, lineType=8, shift=0)
	cv2.imshow('image',image)
	cv2.waitKey(0)
	scores = []
	for itr in range(nimage):
		j_score = jaccard(centers[itr],object_points[itr],image)
		# print(j_score)
		scores.append(j_score)
