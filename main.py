import cv2
import os
import numpy as np

directory = os.fsencode("Data set\Training")

#This function iterates through the images in (argument directory)
#convert each image to greyscale and resize it to 50*50
#then save the result in directory "Data set\\Trainpre\\"
def preprocessing(directory):
    for filename in os.listdir(directory):
        fil = str(os.path.join(directory, filename).decode("utf-8"))
        print(fil)
        image = cv2.imread(fil)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image,(50,50));
        filename = filename.decode("utf-8")
        print(filename)
        cv2.imwrite("Data set\\Trainpre\\"+ filename,gray_image)

def isolateObjects(imgPath):
    # Minimum percentage of pixels of same hue to consider dominant colour
    MIN_PIXEL_CNT_PCT = (1.0/20.0)
    image = cv2.imread(imgPath)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,_,_ = cv2.split(image_hsv)
    bins = np.bincount(h.flatten())
    peaks = np.where(bins > (h.size * MIN_PIXEL_CNT_PCT))[0]
    for i,peak in enumerate(peaks):
        mask = cv2.inRange(h, int(peak), int(peak))
        blob = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite("Data set\\Testpre\\"+"colourblobs-%d-hue_%03d.png" % (i, peak), blob)
