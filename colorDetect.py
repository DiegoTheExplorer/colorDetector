import cv2
import csv
import numpy as np
import os

# image resize while keeping aspect ratio taken from:
# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
# answered by thewaywewere, edited by starball
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def colorDetect(dir):
  files = os.listdir(dir)
  fields = ['Filename', 'Color','Dominant Color Ratio(%)']
  rows = []

  # Original HSV ranges for each color taken from 
  # https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv
  # some values have been changed and pink was added
  
  color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
                'white': [[180, 18, 255], [0, 0, 231]],
                'red2': [[9, 255, 255], [0, 50, 70]],
                'orange': [[20, 255, 255], [10, 50, 70]],
                'yellow': [[35, 255, 255], [21, 50, 70]],
                'green': [[89, 255, 255], [36, 50, 70]],
                'blue': [[114, 255, 255], [90, 50, 70]],
                'purple': [[144, 255, 255], [115, 50, 70]],
                'pink': [[175,255,255], [145, 50, 70]],
                'red1': [[180, 255, 255], [176, 50, 70]],                
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

    # load image and convert to hsv
    path = dir + "/" + filename
    rgb_img = cv2.imread(path)
    width = rgb_img.shape[1]
    height = rgb_img.shape[0]

    # resize if the image is too large
    if (width > 1024):
      rgb_img = image_resize(rgb_img, width=1024)
    if (height > 576):
      rgb_img = image_resize(rgb_img, height=576)
    
    # no post processing version of the image 
    nopp_hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

    # Reflection reduction through inpainting taken from:
    # https://itecnote.com/tecnote/opencv-reflection-reduction-in-image/
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)[1]
    inpaint_img = cv2.inpaint(rgb_img,thresh,3,cv2.INPAINT_TELEA)

    hsv_img = cv2.cvtColor(inpaint_img, cv2.COLOR_BGR2HSV)
    
    # Count the number of pixels for each color
    for key in color_count:
      if (key == 'red'):
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

        if (key == 'white'):
          mask = cv2.inRange(nopp_hsv_img,lower,upper)
        else:
          mask = cv2.inRange(hsv_img,lower,upper)
        color_count[key] = np.count_nonzero(mask)
      
      # color_ratio = (color_count[key] / (rgb_img.shape[0] * rgb_img.shape[1])) * 100
      # print("Percentage of ", key, " pixels: ", color_ratio)
      
    #get the color with the most number of pixels in the image
    car_color = max(color_count, key= lambda x:color_count[x])
    percent_dominant_color_pixels = (color_count[car_color] / (rgb_img.shape[0] * rgb_img.shape[1])) * 100
    rows.append([filename,car_color,percent_dominant_color_pixels])
    # print( " ***************************** The color of the car in " , filename, " is ", car_color, " *****************************")
    

    #display image with the identified color as the window name
    # cv2.imshow(car_color,inpaint_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
  
  with open('data.csv', 'w') as f:
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(rows)
