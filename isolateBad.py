import cv2 as cv2
import os
import numpy as np

directoryChecker = {}
img = cv2.imread('1 (3).png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
for it in range(0,255):
    ret,thresh = cv2.threshold(img,it,255,0)
    im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    for item in range(len(contours)):
        cnt = contours[item]
        gun = len(cnt)
        if len(cnt)>20 and gun not in directoryChecker.keys():
            directoryChecker[len(cnt)] = 0
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x,y,w,h = cv2.boundingRect(cnt)
                print(str(x))
                if x not in directoryChecker.keys():
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
