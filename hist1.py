import cv2
import numpy as np

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''

     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    

    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val


def draw_hist(name, gray, height, width):
   lbp = np.zeros((height, width,3), np.uint8)
   for i in range(0, height):
      for j in range(0, width):
         lbp[i, j] = lbp_calculated_pixel(gray, i, j)
   hist = cv2.calcHist([gray], [0], None, [256], [0,256])
   MAX = max(hist)
   plot = np.zeros((512,1024))
   for i in range(255):
      x1 = 4*i
      x2 = 4*(i+1)
      y1 = hist[i]*512/MAX
      y2 = hist[i+1]*512/MAX
      cv2.line(plot, (x1,y1), (x2,y2), 1, 3)

   gray = cv2.resize(gray, (350, 350))
   lbp = cv2.resize(lbp, (350, 350))
   plot = cv2.resize(plot, (350, 350))
   cv2.imshow("gray", gray)
   cv2.imshow("LBP", lbp)
   cv2.moveWindow("LBP", 400,400)
   cv2.moveWindow("hist", 500,0)
   cv2.imshow("hist", plot)


def main():
    cam = cv2.VideoCapture('video/sample.mp4')
    while cv2.waitKey(10) == -1:
        ret, img = cam.read()
        height, width, channel = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        

        
        draw_hist("cam",gray,height,width)
        cv2.waitKey(10)

if __name__=="__main__":
    main()
