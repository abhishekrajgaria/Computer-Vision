import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
frame = cv2.imread('/content/Cap2.PNG')
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG();    
fgbg2 = cv2.createBackgroundSubtractorMOG2(); 
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG(); 
fgmask1 = fgbg1.apply(frame)
fgmask2 = fgbg2.apply(frame)
fgmask3 = fgbg3.apply(frame)
cv2_imshow(fgmask1)
cv2_imshow(fgmask2)
cv2_imshow(fgmask3)
cv2_imshow(frame)