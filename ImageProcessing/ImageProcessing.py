from PIL.Image import *
import cv2 
dim = (32,32)
img=cv2.imread("ImageProcessing/Test/IMG_8059.png",0) #image a selection
ret3,th3 = cv2.threshold(img,240,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
resized = cv2.resize(th3, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite("ImageProcessing/Result/test.png", resized)


