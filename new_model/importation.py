from keras.preprocessing.image import ImageDataGenerator
import pathlib
import tensorflow as tf
import numpy as np

def extract(path):
    data_dir = pathlib.Path("DATA")
    image_count = len(list(data_dir.glob('*/*')))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]) 
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # redimension des images
    BATCH_SIZE = image_count
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        classes = list(CLASS_NAMES))

    img,label  = next(train_data_gen)

    return img,label,CLASS_NAMES