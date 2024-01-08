from colorDetect import *
import cv2
import numpy as np
import os

def colorDetect(dir):
  files = os.listdir(dir)
  # HSV ranges for each color taken from 
  # https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv

  color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 1]],
                'white': [[180, 18, 255], [0, 0, 231]],
                'red1': [[180, 255, 230], [166, 50, 70]],
                'red2': [[9, 255, 230], [0, 50, 70]],
                'green': [[89, 255, 230], [36, 50, 70]],
                'blue': [[128, 255, 230], [90, 50, 70]],
                'yellow': [[35, 255, 230], [25, 50, 70]],
                'purple': [[144, 255, 230], [129, 50, 70]],
                'pink': [[165,255,230], [145, 50, 70]],
                'orange': [[24, 255, 230], [10, 50, 70]],
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
    height = rgb_img.shape[0]
    width = rgb_img.shape[1]

    mask = np.zeros(rgb_img.shape[:2],np.uint8)
    bgdModel =  np.zeros((1,65),np.float64)
    fgdModel =  np.zeros((1,65),np.float64)

    rect =	(0,0,width - 1, height - 1)
    cv2.grabCut(rgb_img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 =	np.where((mask==2)|(mask==0),0,1).astype('uint8')
    rgb_img   =	rgb_img*mask2[:,:,np.newaxis]
    cv2.imshow("after grabcut", rgb_img)
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

