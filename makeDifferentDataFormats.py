import numpy as np
import os
import cv2

"""
From 299x299 rgb files, I want to create the following other data sets in "Training Data"
(1) 299x299 gray
(2) 200x200 rgb
(3) 200x200 gray
(4) 100x100 rgb
(5) 100x100 gray
"""

# For each file in training data,

# Open, get image in a format where Imshow works.
# Use cv2 to convert to gray for  (1)
# Use cv2 to rescale to 200x200 - > (2)
# gray_scale  -> (3)
#...


def create_data_sets(Dtype):
    training_fp = "trainingData/rgb_299/{}".format(Dtype)

    for root, dirs, files in os.walk(training_fp):
        for fname in files:
            print(fname)
            training_data = np.load(os.path.join(training_fp,fname))
            gray_299 = []
            rgb_224 = []
            bgr_224 = []
            rgb_200 = []
            gray_200 = []


            for data in training_data:

                image = data[0]
                label = data[1]
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                gray_299.append([gray_image, label])

                image224 = cv2.resize(image,(224,224))
                bgr_image224 = cv2.cvtColor(image224, cv2.COLOR_RGB2BGR)
                rgb_224.append([image224, label])
                bgr_224.append([bgr_image224, label])


                image200 = cv2.resize(image,(200,200))
                rgb_200.append([image200, label])
                gray_image = cv2.cvtColor(image200, cv2.COLOR_RGB2GRAY)
                gray_200.append([gray_image, label])


            # This could also be a dict of some sort, but lists work.
            dataLst = [gray_299, rgb_224, bgr_224, rgb_200, gray_200]
            folder_names =["gray_299", "rgb_224", "bgr_224", "rgb_200", "gray_200"]

            for i in range(len(folder_names)):
                np.save("trainingData/{}/{}/{}".format(folder_names[i], Dtype, fname), dataLst[i])

#create_data_sets('body')
create_data_sets('head')
