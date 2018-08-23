import numpy as np
from PIL import Image
from Xlib import display, X

def grabscreen(root,W,H):
    raw = root.get_image(0,0, W, H, X.ZPixmap, 0xffffffff)
    image = np.array(Image.frombytes("RGB", (W,H), raw.data, "raw", "RGBX"))
    return image
