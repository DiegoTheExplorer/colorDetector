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
    rgb_img = cv2.resize(rgb_img,(0, 0), fx = 0.5, fy = 0.5)
    rgb_img = cv2.blur(rgb_img,(10,10))
    cv2.imshow("resized",rgb_img)
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

