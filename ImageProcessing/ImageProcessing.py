import cv2
import numpy as np

src = cv2.imread("ImageProcessing/Test/IMG_8065.png")

bg = cv2.dilate(src, np.ones((5,5), dtype=np.uint8))
bg = cv2.GaussianBlur(bg, (5,5), 1)
src_no_bg = 255 - cv2.absdiff(src, bg)
maxValue = 255
thresh = 160
dim = (src.shape[1] - 15, src.shape[0] - 100 )
retval, dst = cv2.threshold(src_no_bg, thresh, maxValue, cv2.THRESH_BINARY_INV)
ret3,th3 = cv2.threshold(dst,thresh,maxValue,cv2.THRESH_BINARY)
cv2.imwrite("ImageProcessing/Result/test.png", th3)
