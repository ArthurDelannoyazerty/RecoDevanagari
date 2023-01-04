import cv2
import numpy as np

src = cv2.imread("ImageProcessing/Test/IMG_8065.png")
bg = cv2.dilate(src, np.ones((5,5), dtype=np.uint8))
bg = cv2.GaussianBlur(bg, (5,5), 1)
src_no_bg = 255 - cv2.absdiff(src, bg)
gray1 = cv2.cvtColor(src_no_bg, cv2.COLOR_BGR2GRAY)

maxValue = 255
thresh = 160
dim = (src.shape[1] - 15, src.shape[0] - 100 )
retval, dst = cv2.threshold(gray1, thresh, maxValue, cv2.THRESH_BINARY_INV)
ret3,th3 = cv2.threshold(dst,thresh,maxValue,cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img_contours = np.zeros(src.shape)
# cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
dilation = cv2.dilate(th3, rect_kernel, iterations = 1)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
im2 = th3.copy()
i = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
     
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cropped = im2[y:y + h, x:x + w]
    cv2.imwrite("ImageProcessing/Result/ImageCropped/Cropped_"+str(i)+".png",cropped)
    i = i+1
    # Open the file in append mode     
    # Apply OCR on the cropped image

cv2.imwrite("ImageProcessing/Result/test.png", im2)
