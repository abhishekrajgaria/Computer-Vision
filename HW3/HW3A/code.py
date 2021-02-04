import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
video = cv2.VideoCapture("denis_walk.avi")


cnt = 0
shape = 0
frames = []
while(video.isOpened()):
	ret, frame = video.read()
	if ret == False:
		break

	cv2.imwrite('D:/course_stuff/CV/HW3/extframe/frame'+str(cnt)+'.jpg',frame)
	cnt+=1
	frames.append(frame)
	# shape = frame.shape

video.release()
cv2.destroyAllWindows()

row = len(frames[0])
col = len(frames[0][0])



# print(background.shape)

# for frame in frames:
# 	for i in range(row):
# 		for j in range(col):
# 			background[i][j][0] += frame[i][j][0]
# 			background[i][j][1] += frame[i][j][1]
# 			background[i][j][2] += frame[i][j][2]
	
# for i in range(row):
# 	for j in range(col):
# 		background[i][j][0] = background[i][j][0]//68
# 		background[i][j][1] = background[i][j][1]//68
# 		background[i][j][2] = background[i][j][2]//68
# 		print(background[i][j][0])
# 		break


# background/=cnt

# mode_s = stats.mode(frames)
# print(mode_s[0][0])
rbackground = np.zeros(frames[0].shape)
for i in range(row):
	for j in range(col):
		temp_arr1 = [0]*256
		temp_arr2 = [0]*256
		temp_arr3 = [0]*256
		for k in range(cnt):
			temp_arr1[frames[k][i][j][0]]+=1
			temp_arr2[frames[k][i][j][1]]+=1
			temp_arr3[frames[k][i][j][2]]+=1
		t_max = 0
		for tt in range(256):
			if(temp_arr1[tt]>t_max):
				t_max = temp_arr1[tt]
				rbackground[i][j][0] = tt
		t_max = 0
		for tt in range(256):
			if(temp_arr2[tt]>t_max):
				t_max = temp_arr2[tt]
				rbackground[i][j][1] = tt
		t_max = 0
		for tt in range(256):
			if(temp_arr3[tt]>t_max):
				t_max = temp_arr3[tt]
				rbackground[i][j][2] = tt




# exit(0)
# rbackground = np.median(frames, axis=0)#.astype(dtype=np.uint8)
background = rbackground.astype(dtype=np.uint8)
cv2.imshow('image',background)
cv2.waitKey(0)
# plt.imshow(background)

new_frames = []
itt = 0
for frame in frames:
	print(itt)
	itt+=1
	new_frame = np.absolute(frame - rbackground)
	new_frame = cv2.cvtColor(new_frame.astype(dtype=np.uint8), cv2.COLOR_BGR2GRAY)
	# new_frame = new_frame/np.amax(new_frame)
	# new_frame
	# t_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
	min_o = -1
	thresh = -1
	for t in range(0,256,4):
		w0 = 0
		w1 = 0
		pl = []

		ph = []
		for i in range(row):
			for j in range(col):
				if(new_frame[i][j]<t):
					w0+=1
					pl.append(new_frame[i][j])
				elif(new_frame[i][j]>=t):
					w1+=1
					ph.append(new_frame[i][j])

		if(len(pl)==0):
			v0 = 0
		else:
			v0 = np.var(np.array(pl))
		
		if(len(ph)==0):
			v1 = 0
		else:
			v1 = np.var(np.array(ph))

		o =w0*v0 + w1*v1
		if(min_o<0):
			min_o = o
			thresh = t
		else:
			if(min_o > o ):
				min_o = o
				thresh = t
		# print(thresh)
	# print(thresh)
	image1 = np.zeros(new_frame.shape)
	image2 = np.zeros(new_frame.shape)

	# thresh = 93

	cnt1 = 0
	cnt2 = 0
	for i in range(row):
		for j in range(col):
			if(new_frame[i][j]<thresh):
				image1[i][j] = 255
				cnt1+=1
			else:
				cnt2+=1
				image2[i][j] = 255

	# print(cnt1,cnt2)
	# cv2.imshow('image', image1)
	# cv2.waitKey(0)
	# cv2.imshow('image', image2)
	# cv2.waitKey(0)
	fg = 0
	bg = 0
	if(cnt1<cnt2):
		fg = image1
		bg = image2
		# cv2.imshow('image',image1)
		# cv2.waitKey(0)
	else:
		fg = image2
		bg = image1
		# cv2.imshow('image',image2)
		# cv2.waitKey(0)
	new_frames.append(fg)
	# new_frame[new_frame > thresh] = 255
	# new_frame[new_frame <= thresh] = 0
	# new_frames.append(new_frame)
	# break
	

cv2.imshow('image',new_frames[0])
cv2.waitKey(0)

cv2.imshow('image',new_frames[5])
cv2.waitKey(0)

cv2.imshow('image',new_frames[10])
cv2.waitKey(0)
for j in range(cnt):
	cv2.imwrite('D:/course_stuff/CV/HW3/modeframe/frame'+str(j)+'.jpg',new_frames[j])


