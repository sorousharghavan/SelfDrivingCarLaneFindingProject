import math
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#Variables to store averaging data
lane_history = []
avg_lane = [[[0,0],[0,0]],[[0,0],[0,0]]]
#We dont want to show the averaged data until we have enough. So we have this counter to count the number of data points
#until we have enough
average_initializer = 0

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    #for line in lines:
    #    for x1,y1,x2,y2 in line:
    #        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    try:
        draw_lines2(img, extrapolate(img.shape[0], update_lanes(img, lines)))
    except:
        print("division by zero")

def draw_lines2(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        x1=line[0][0]
        x2=line[1][0]
        y1=line[0][1]
        y2=line[1][1]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def extrapolate(img_height, lanes):
    for lane in lanes:
        t = lane[0]
        b = lane[1]
        x_ex = (int)(t[0] + (img_height - t[1])/(b[1]-t[1]) * (b[0] - t[0]))
        lanes[lanes.index(lane)][1] = [x_ex, img_height]
    return lanes

def update_lanes(image, new_lines):
    '''Determines whether a line belongs to the right lane or left lane
    Adds the new lane to the corresponding averaged dataset'''
    height = image.shape[0]
    width = image.shape[1]
    midx = width/2
    
    miny_r = height
    minx_r = width
    maxy_r = 0
    maxx_r = 0
    
    miny_l = height
    minx_l = width
    maxy_l = 0
    maxx_l = 0
    for line in new_lines:
        for x1,y1,x2,y2 in line:
            if (x1 >= midx or x2 >= midx):
                if y1 <= miny_r:
                    miny_r = y1
                    minx_r = x1
                if y1 >= maxy_r:
                    maxy_r = y1
                    maxx_r = x1
                if y2 <= miny_r:
                    miny_r = y2
                    minx_r = x2
                if y2 >= maxy_r:
                    maxy_r = y2
                    maxx_r = x2
            else:
                if y1 <= miny_l:
                    miny_l = y1
                    minx_l = x1
                if y1 >= maxy_l:
                    maxy_l = y1
                    maxx_l = x1
                if y2 <= miny_l:
                    miny_l = y2
                    minx_l = x2
                if y2 >= maxy_l:
                    maxy_l = y2
                    maxx_l = x2
    return update_avg([[[minx_r,miny_r],[maxx_r, maxy_r]], [[minx_l, miny_l], [maxx_l, maxy_l]]])

def update_avg(new_lane):
    '''Updates the averaged lanes with a new dataset'''
    global lane_history
    global avg_lane
    global average_initializer
    
    if average_initializer == 30:
        lane_history.remove(lane_history[0])
        lane_history.append(new_lane)
    else:
        lane_history.append(new_lane)
        average_initializer = average_initializer + 1
    for lane in avg_lane:
        for end in lane:
            for point in end:
                l_i = avg_lane.index(lane)
                e_i = lane.index(end)
                p_i = end.index(point)
                avgtemp = None
                for item in lane_history:
                    if avgtemp == None:
                        avgtemp = item[l_i][e_i][p_i]
                    else:
                        avgtemp = (int)((avgtemp * lane_history.index(item) + item[l_i][e_i][p_i])/(lane_history.index(item)+1))
                avg_lane[l_i][e_i][p_i] = avgtemp
    return avg_lane


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def show_image(img):
    """Displays image"""
    plt.imshow(img)
    plt.show()

#Pipeline
def process_image(image):
    '''Main pipeline to process image and draw overlay lanes on the image
    returns the final image'''
    height = image.shape[0]
    width = image.shape[1]

    gray = grayscale(image)
        
    gray_blurred = gaussian_blur(gray, 5)

    #show_image(gray_blurred)
        
    gray_canny = canny(gray_blurred, 150, 230)

    #show_image(gray_canny)
        
    mask_region = np.array([[[(int)(0.2*width), (int)(0.9*height)],[(int)(0.43*width), (int)(0.63*height)],[(int)(0.58*width),(int)(0.63*height)],[(int)(0.9*width), (int)(0.9*height)]]])
       
    masked_img = region_of_interest(gray_canny, mask_region)

    #show_image(masked_img)
        
    masked_hough = hough_lines(masked_img, 1, np.pi/180, 20, 50, 100)

    #show_image(masked_hough)

    #show_image(weighted_img(masked_hough, image))
    
    return weighted_img(masked_hough, image)



######################################################

#Deprecated
#Method for manually finding x1,x2,y1,y2 to extrapolate
def extrapolate_lanes(img):
    return extrapolate(find_lane_coordinates(img))


def find_lane_coordinates(img):
    mask_left = np.array([[[470,0], [960,0], [960,540], [470,540]]])
    mask_right = np.array([[[470,0], [470,540], [0,540], [0,0]]])
    
    right = region_of_interest(img, mask_left)
    left = region_of_interest(img, mask_right)
    #show_image(right)
    #show_image(left)
    
    rows = right.shape[1]
    cols = right.shape[0]

    flag = False
    lines = []
    line = []
    for y in range(320,rows):
        for x in range(200,cols):
            if not flag and np.array_equal(right[x,y], [255,0,0]):
                flag = True
                line.append([y,x])
                break

    flag = False
    for y in range(rows-1,0,-1):
        for x in range(0,cols):
            if not flag and np.array_equal(right[x,y], [255,0,0]):
                flag = True
                line.append([y,x])
                break
                
    lines.append(line)

    flag = False
    rows = left.shape[1]
    cols = left.shape[0]
    line = []
    for y in range(rows-1,0,-1):
        for x in range(200,cols):
            if not flag and np.array_equal(left[x,y], [255,0,0]):
                flag = True
                line.append([y,x])
                break
                
    flag = False
    for y in range(320,rows):
        for x in range(0,cols):
            if not flag and np.array_equal(left[x,y], [255,0,0]):
                flag = True
                line.append([y,x])
                break
    
    lines.append(line)
    print("detected lane coordinates : \n", lines)
    return lines
