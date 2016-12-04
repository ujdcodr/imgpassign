import numpy as np
import cv2
import imutils

image=cv2.imread('jellyfish.jpg') # reading image
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert to grayscale
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# Adaptive Gaussian Threshold works best given the variable lighting conditions
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 1)

# removing false positives
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(thresh,kernel,iterations = 2)
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

count=0 # counting no. of jellyfish
# find contours in the thresholded image
cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    if M["m00"]!= 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX=0
        cY=0
    # draw the contour and center of the shape on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    # applying the "red cross" at the centroid 
    cv2.putText(image, "X", (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    count+=1 # increment no. of jellyfish

cv2.putText(image, "No. of Jellyfish:"+str(count), (0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
