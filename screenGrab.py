import cv2, time
import numpy as np
from PIL import Image
from Xlib import display, X


def process_img(image):
    #convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    #processed_img = cv2.Canny(process_img, threshold1=200, threshold2=300)
    return processed_img

def grabscreen(root,W,H):
    raw = root.get_image(0,0, W, H, X.ZPixmap, 0xffffffff)
    image = np.array(Image.frombytes("RGB", (W,H), raw.data, "raw", "RGBX"))
    #image = process_img(image)
    #cv2.imshow('window',image)
    return image
