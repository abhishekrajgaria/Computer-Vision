import cv2
import numpy as np
# from collection import deq
limit = 200000
import sys
sys.setrecursionlimit(limit)


def isvalid(img, i, j, visited):
	row = len(img)
	col = len(img[0])
	return (((i>=0) and i<row and j>=0 and j<col and (img[i][j]>0 and visited[i][j]==0)))
	# return False


def dfs(img, i, j, visited):
	rowmov = [-1,-1,-1,0,0,1,1,1]
	colmov = [-1,0,1, -1, 1, -1, 0, 1]
	stack = []
	stack.append([i,j])
	while(stack != []):
		x = stack.pop()
		r = x[0]
		c = x[1]
		visited[r][c] = 1
		for k in range(8):
			if(isvalid(img, r+rowmov[k], c+colmov[k],visited)):
				stack.append([r+rowmov[k],c+colmov[k]])

def count_image(imgmat):

	row = len(imgmat)
	col = len(imgmat[0])
	visited = np.zeros(imgmat.shape)

	count = 0

	for i in range(row):
		for j in range(col):
			if(imgmat[i][j]>0 and visited[i][j]==0):
				
				dfs(imgmat,i,j,visited)
				count+=1

	# print("hello")
	# print(count)
	return count


if __name__ == '__main__':
		

	image = cv2.imread('binary_image.png')
	# cv2.imshow('image',image)
	# cv2.waitKey(0)
	print(image.shape)

	r_layer = np.zeros((len(image),len(image[0])))
	g_layer = np.zeros((len(image),len(image[0])))
	b_layer = np.zeros((len(image),len(image[0])))


	for i in range(len(image)):
		for j in range(len(image[i])):
			r_layer[i][j] = image[i][j][0]
			g_layer[i][j] = image[i][j][1]
			b_layer[i][j] = image[i][j][2]

	print(count_image(r_layer))
