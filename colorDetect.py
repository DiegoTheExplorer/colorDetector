from colorDetect import *
import cv2
import numpy as np
import os

def colorDetect(dir):
  files = os.listdir(dir)
  # HSV ranges for each color taken from 
  # https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv

  color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
                'white': [[180, 18, 255], [0, 0, 231]],
                'red1': [[180, 255, 255], [166, 50, 70]],
                'red2': [[9, 255, 255], [0, 50, 70]],
                'green': [[89, 255, 255], [36, 50, 70]],
                'blue': [[128, 255, 255], [90, 50, 70]],
                'yellow': [[35, 255, 255], [25, 50, 70]],
                'purple': [[144, 255, 255], [129, 50, 70]],
                'pink': [[165,255,255], [145, 50, 70]],
                'orange': [[24, 255, 255], [10, 50, 70]],
                'gray': [[180, 18, 230], [0, 0, 40]]}

  color_count = {'black': 0,
                'white': 0,
                'red': 0,
                'green': 0,
                'blue': 0,
                'yellow': 0,
                'purple': 0,
                'pink':0,
                'orange': 0,
                'gray': 0}

  for filename in files:

    #load image and convert to hsv
    path = "images/" + filename
    rgb_img = cv2.imread(path)

    # Color Quantization taken from 
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html#kmeans-opencv
    Z = rgb_img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((rgb_img.shape))

    cv2.imshow('res2',res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    #Count the number of pixels for each color
    for key in color_count:
      if(key == 'red'):
        upper = tuple(color_dict_HSV['red1'][0])
        lower = tuple(color_dict_HSV['red1'][1])

        mask = cv2.inRange(hsv_img,lower,upper)
        color_count[key] = np.count_nonzero(mask)

        upper = tuple(color_dict_HSV['red2'][0])
        lower = tuple(color_dict_HSV['red2'][1])

        mask = cv2.inRange(hsv_img,lower,upper)
        color_count[key] += np.count_nonzero(mask)
      else:
        upper = tuple(color_dict_HSV[key][0])
        lower = tuple(color_dict_HSV[key][1])

        mask = cv2.inRange(hsv_img,lower,upper)
        color_count[key] = np.count_nonzero(mask)
      
    #get the color with the most number of pixels in the image
    car_color = max(color_count, key= lambda x:color_count[x])
    print("The color of the car in " , filename, " is ", car_color)

