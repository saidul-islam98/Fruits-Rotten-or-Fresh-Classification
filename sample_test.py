import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt


def simple_test(BASE, class_labels, cnn_model):
    test_dir = os.path.join(BASE, 'sample_test')
    for i, data in enumerate(os.listdir(test_dir)):
        path = os.path.join(test_dir, data)
        print(path)
        image = Image.open(path)
        # summarize some details about the image
        print(image.format)
        print(image.size)
        print(image.mode)
        # show the image

        # display the array of pixels as an image
        plt.imshow(image)
        plt.show()
        img = load_img(path, target_size=(228,228,3))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = cnn_model.predict(x)
        print("Overall prediction: ")
        print(pred)
        maxidx = np.argmax(max(pred))
        for k in class_labels.items():
            #print(k[0])
            if k[1]==maxidx:
                print("Class name:", k[0])
                break
        print(max(pred))