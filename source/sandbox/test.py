import numpy as np
import os
IMAGE_DIR = "../../data"
directory = os.fsencode(IMAGE_DIR)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        print(IMAGE_DIR + '/' + filename)