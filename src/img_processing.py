import numpy as np
import imageio
import cv2
import sys, os

#Processing Original Image
def process_img(location_img):
    image = imageio.imread(location_img)
    image = image.astype(np.float32)/255 

    return image

#Load and construct Ground Truth
def read_gt(location_gt):    
    entries = os.listdir(location_gt)
    gt_images = []

    #Collect all human labelled images
    for entry in entries:
        ground_truth = imageio.imread(location_gt+entry)
        ground_truth = ground_truth.astype(np.float64)/255 
        gt_images.append(ground_truth)

    return gt_images

#Construct Ground Truth representation from all human labelled images
def construct_gt(location_gt):
    gt_images = read_gt(location_gt)
    size = gt_images[0].shape[:2]

    pixels = np.zeros((size))

    for k in range(len(gt_images)):
        ret, bw_img = cv2.threshold(gt_images[k],0.0001,1,cv2.THRESH_BINARY)   
        for i in range(size[0]):
            for j in range(size[1]):            
                if(bw_img[i,j][0]>0 and bw_img[i,j][1]==0 and bw_img[i,j][2]==0):
                    pixels[i][j] += 1   

    #Each pixel is in foreground if N-1 out of N humans labelled the pixel in the foreground, else in the background
    pixels = np.where(pixels >=len(gt_images)-1, 1., 0.)

    F = len(np.where(pixels>0)[0])
    B = len(np.where(pixels==0)[0])

    print("Foreground area of constructed Ground Truth is %d pixels"% F)
    print("Background area of constructed Ground Truth is %d pixels\n"% B)

    return pixels, F
