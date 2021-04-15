import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image


BASE = './data'

TRAINING_DIR = os.path.join(BASE, 'train')

fapple = os.path.join(TRAINING_DIR, "freshapples")
freshapples = os.listdir(fapple)
for i in range(0,5):
  im = os.path.join(fapple,freshapples[i])
  image = Image.open(im)
  # summarize some details about the image
  print(image.format)
  print(image.size)
  print(image.mode)
  # show the image

  # display the array of pixels as an image
  plt.imshow(image)
  plt.show()