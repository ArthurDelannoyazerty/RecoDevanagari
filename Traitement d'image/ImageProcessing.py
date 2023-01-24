import cv2
import numpy as np
#import pytesseract

def img_bgr_2_gray():
    
    src = cv2.imread("ImageProcessing/Test/chtxt.png")
    bg = cv2.dilate(src, np.ones((5,5), dtype=np.uint8))
    bg = cv2.GaussianBlur(bg, (5,5), 1)
    src_no_bg = 255 - cv2.absdiff(src, bg)
    gray1 = cv2.cvtColor(src_no_bg, cv2.COLOR_BGR2GRAY)

    maxValue = 255
    thresh = 160
    dim = (src.shape[1] - 15, src.shape[0] - 100 )
    retval, dst = cv2.threshold(gray1, thresh, maxValue)
    ret3,th3 = cv2.threshold(dst,thresh,maxValue,cv2.THRESH_BINARY)


    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    dilation = cv2.dilate(th3, rect_kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
    im2 = th3.copy()
    i = 0
    for contour in contours:
        # Find the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        
        # Crop the image
        crop_image = im2[y:y+h, x:x+w]
        
        # Save the cropped image to a file
        cv2.imwrite("reco中文/ImageProcessing/Result/ImageCropped/Cropped_{}.png".format(i), crop_image)
            
        cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)

        i += 1

    cv2.imwrite("reco中文/ImageProcessing/Result/test.png", im2)

