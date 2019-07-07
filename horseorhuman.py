import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

horses = load_images_from_folder('horse-or-human/horses')

for img in horses:
    cv2.imshow()
