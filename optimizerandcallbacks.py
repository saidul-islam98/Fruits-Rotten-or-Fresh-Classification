import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os


def checkpoint_model(CHECKPOINT_BASE):
    checkpoint_filepath = os.path.join(CHECKPOINT_BASE, "model_checkpoint")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               save_weights_only=True,
                                                               monitor='val_accuracy',
                                                               mode='max',
                                                               save_best_only=True)
    return checkpoint_filepath, model_checkpoint_callback


def lr_reducer():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                verbose=0, mode='auto', min_delta=0.0001,
                                cooldown=0, min_lr=0)
    return reduce_lr

