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

def min_circle(ps):
	# print("youl")
	# assert(len(ps)<=3)
	# print("youl67")
	if(len(ps)<=3):
		if(len(ps)==0):
			return Circle(Point(0,0),0)
		elif(len(ps)==1):
			return Circle(ps[0],0)
		elif(len(ps)==2):
			return gen_circle2(ps[0],ps[1])

		for i in range(3):
			for j in range(i+1,3):
				c = gen_circle2(ps[i],ps[j])
				if(is_circle(c,ps)):
					return c
		return gen_circle3(ps[0],ps[1],ps[2])
	return Circle(Point(0,0),0)

def util(ps, rs, n):
	print("----------------")
	for pps in range(n):
		print(ps[pps].x, ps[pps].y,end=" /")
	print()
	# print(ps)
	# print()
	# print(rs)
	# print()
	print(n,len(rs))
	print("=====================")
	if(n==0 or len(rs)==3):
		# print("hello4")
		return min_circle(rs)

	ind = random.randint(0,n-1)
	p = ps[ind]
	print(ind)
	ps[ind] = ps[n-1]
	ps[n-1] = p
	for pps in range(n):
		print(ps[pps].x, ps[pps].y,end=" /")
	print()
	d = util(ps,rs,n-1)
	print("disco")
	if(inside(d,p)):
		print("hello2")
		return d

	rs.append(p)
	return util(ps,rs,n-1)

def create_points(arr):
	ps = []
	for i in arr:
		ps.append(Point(i[0],i[1]))
	return ps

def circlefunc(arr):
	ps = create_points(arr)
	pps = ps.copy()
	random.shuffle(pps)
	print("hello1")
	print(len(pps))
	return util(pps,[],len(pps))


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
				# print(count)
				# print(len(point_set))

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

def naive_app(pts):
	maxd = 0
	pt1 = 0
	pt2 = 0
	ppt = create_points(pts)
	for pt in ppt:
		for ptt in ppt:
			temp = dist(pt,ptt)
			if(maxd<temp):
				pt1 = pt
				pt2 = ptt
				maxd = temp
	r = maxd//2
	x = (min(pt1.x,pt2.x) + abs(pt1.x-pt2.x))
	y = (min(pt1.y,pt2.y) + abs(pt1.y-pt2.y))
	return Circle(Point(x,y),r)

def naive_app2(pts):
	maxd = 0
	pt1 = 0
	pt2 = 0
	ppt = create_points(pts)
	for pt in ppt:
		for ptt in ppt:
			temp = dist(pt,ptt)
			if(maxd<temp):
				pt1 = pt
				pt2 = ptt
				maxd = temp
	r = maxd//2
	x = (min(pt1.x,pt2.x) + abs(pt1.x-pt2.x))
	y = (min(pt1.y,pt2.y) + abs(pt1.y-pt2.y))

	mean_pt = Point(0,0)
	for pt in pts:
		mean_pt.x += pt[0]
		mean_pt.y += pt[1]

	mean_pt.x /= len(pts)
	mean_pt.y /= len(pts)
	r1 = 0
	ppt = create_points(pts)
	for pt in ppt:
		temp = dist(pt,mean_pt)
		if(r1<temp):
			r1 = temp
	return Circle(mean_pt,r)


if __name__ == '__main__':
		

	image = cv2.imread('binary_image.png')
	image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	# cv2.circle(image, (250,250), 100, 255, thickness=1, lineType=8, shift=0)

	# cv2.circle(image, (450,450), 100, 255, thickness=1, lineType=8, shift=0)
	# cv2.imshow('image',image)
	# cv2.waitKey(0)
	print(image.shape)
	# cv2.imshow('image',image)
	# cv2.waitKey(0)
	# fg_hist = [0]*256
	# for i in range(len(image)):
	# 	for j in range(len(image[0])):
	# 		fg_hist[image[i][j]] += 1

	# plt.plot(fg_hist)
	# plt.show()

	nimage, object_points = count_image(image)
	print(nimage)
	custom = [[100,100],[150,150],[200,200],[400,400],[600,600]]
	for itr in range(nimage):
		# print(object_points[itr][:50])
		hull = ConvexHull(object_points[itr])
		points = hull_points(object_points[itr],hull)
		# print(points)
		# print(len(points))
		c = brute_circ(points)
		# c = circlefunc(points)
		# c = naive_app(points)
		# c = naive_app2(points)
		print(c.c.x,c.c.y,c.r)
		cv2.circle(image, (math.floor(c.c.y),math.floor(c.c.x)), math.floor(c.r), 255, thickness=1, lineType=8, shift=0)
		cv2.imshow('image',image)
		cv2.waitKey(0)
		# break

	# hull1 = ConvexHull(object_points[0])
	# print(len(hull1.vertices))
	# print(hull1.vertices[0])
	# c = circlefunc(hull1)
	# print(c.c.x,c.c.y,c.r)
	# print("yahallo!")
	

	# print(count_image(r_layer))
