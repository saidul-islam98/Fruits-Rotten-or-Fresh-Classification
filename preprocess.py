import PIL
import os
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

def train_validation_datagen(BASE):
    TRAINING_DIR = os.path.join(BASE, "train")
    training_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = os.path.join(BASE, "test")
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(228,228),
        class_mode='categorical',
    batch_size=64
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(228,228),
        class_mode='categorical',
    batch_size=64
    )

    class_labels = train_generator.class_indices

    return train_generator, validation_generator, class_labels