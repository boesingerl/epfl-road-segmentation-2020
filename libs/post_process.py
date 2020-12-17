import cv2
import numpy as np

def morphological(img, morph_type,kernel_size=20):
  """Applies the given cv2 morphological operations to the image, using column, row, and square kernels sequentially"""
  img = img.astype('uint8')
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,kernel_size))
  closed = cv2.morphologyEx(img, morph_type, kernel )

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,1))
  closed = cv2.morphologyEx(closed, morph_type, kernel )

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
  closed = cv2.morphologyEx(closed, morph_type, kernel )
  return closed

def open_pred(img, kernel_size=20):
  """Applies the cv2.MORPH_OPEN operation to the image, using morphological"""
  return morphological(img, cv2.MORPH_OPEN, kernel_size=kernel_size)
  
def close_pred(img, kernel_size=20):
  """Applies the cv2.MORPH_CLOSE operation to the image, using morphological"""
  return morphological(img, cv2.MORPH_CLOSE, kernel_size=kernel_size)
  
def smooth_predictions(pred, steps=5, kernel_size=31):
  """Applies Gaussian smoothing, for multiple steps in a row, with the given kernel size"""
  for i in range(steps):
    pred = cv2.GaussianBlur(pred,(kernel_size,kernel_size),cv2.BORDER_DEFAULT)
  return pred

def hough_find_lines(pred, rho=1, theta=np.pi/180, threshold=70, min_line_length=50, max_line_gap=50):
  """Returns all found lines using the Hough transform for lines and given parameters"""
  img = np.zeros_like(pred)
  edges = cv2.Canny(pred,20,150,apertureSize = 3)
  # Run Hough on edge detected image
  # Output "lines" is an array containing endpoints of detected line segments
  lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                      min_line_length, max_line_gap)
  if lines is not None:
    for line in lines:
      if line is not None:
        for x1,y1,x2,y2 in line:
          cv2.line(img,(x1,y1),(x2,y2),(255,0,0),10)
  return img

def keep_large_area(img,min_area=2000,max_area=9999999999):
  """Keeps only blobs of large enough area, with threshold as argument"""
  ret, thresh = cv2.threshold(img, 127, 255, 0)
  contours,h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  results = []
  for c in contours:
    result = np.zeros_like(img)
    area = cv2.contourArea(c)
    #only keep important areas (large enough)
    if area > min_area and area < max_area:
        cv2.drawContours(result, [c], 0, 255, -1)
        results.append(np.asarray(result))
  #apply modulo operation to not fill up hollow areas inside big areas
  return np.sum(np.stack(results),axis=0) % 510
