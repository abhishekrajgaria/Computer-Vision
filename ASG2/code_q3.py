
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

c_image = cv2.imread("contrast_cue.png")
c_image = cv2.cvtColor(c_image,cv2.COLOR_RGB2GRAY)
print(c_image.shape)
s_image = cv2.imread("spatial_cue.png")
s_image = cv2.cvtColor(s_image,cv2.COLOR_RGB2GRAY)
print(s_image.shape)


def z_value(mf,sf,mb,sb,flag):
	term1 = ((mb*pow(sf,2) - mf*pow(sb,2))/(pow(sf,2)-pow(sb,2)))
	term2 = ((sf*sb)/(pow(sf,2)-pow(sb,2)))
	term3 = pow((mf-mb),2) - (2*(pow(sf,2) - pow(sb,2))*(np.log(sb) - np.log(sf)))
	if(flag==1):
		return term1 + (term2*np.sqrt(term3))
	else:
		return term1 - (term2*np.sqrt(term3))

def separation_score(timage):
	m,n = timage.shape[:2]

	th_value,th_image = cv2.threshold(timage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	print("th_value", th_value)
	# print(th_image)
	print("th_image shape",th_image.shape)
	image = np.float32(timage)/np.float32(np.amax(timage))

	fpx = []
	bpx = []

	print(m,n)
	# exit(0)
	for i in range(m):
		for j in range(n):
			if(th_image[i][j]==0):
				bpx.append(image[i][j])
			else:
				fpx.append(image[i][j])
	# print(bpx)
	bpx = np.array(bpx)
	fpx = np.array(fpx)
	mf = np.mean(fpx)# + 1.0
	mb = np.mean(bpx)# + 1.0
	sf = np.std(fpx) + 1e-5
	sb = np.std(bpx) + 1e-5
	print(mf, sf,"---",mb,sb)
	z1 = z_value(mf,sf,mb,sb,1)#/255.0
	z2 = z_value(mf,sf,mb,sb,0)#/255.0
	
	gamma = 256
	print(z1,z2)
	L = 0
	f3 = lambda z: 	np.exp(-(np.square((z - mf)/sf)))/(sf*(np.sqrt(2*np.pi)))

	f4 = lambda z: 	np.exp(-(np.square((z - mb)/sb)))/(sb*(np.sqrt(2*np.pi)))

	if(z1<=1):
		print("z1")
		L = quad(f3, 0,z1)[0] + quad(f4,z1,1)[0]

	else:
		L = quad(f3, 0,z2)[0] + quad(f4,z2,1)[0]

	print("L",L)

	phi = 1/(1+np.log10(1 + gamma*(L)))

	print(phi)

	return phi


cscore = (separation_score(c_image))
print("---------------------------------")
sscore = (separation_score(s_image))

c_image = np.float32(c_image)/np.float32(np.amax(c_image))
s_image = np.float32(s_image)/np.float32(np.amax(s_image))

saliency_image = c_image*cscore + s_image*sscore
saliency_image = saliency_image/np.amax(saliency_image)

saliency_image = saliency_image*255.0

saliency_image = np.uint8(saliency_image)
plt.imshow(saliency_image,cmap="gray")
plt.show()
cv2.imwrite("saliency_image.png", saliency_image)