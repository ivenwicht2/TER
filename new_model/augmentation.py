from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def augment(image,label):
    new_image = image
    new_label = label
    train_datagen = []
    train_datagen.append(ImageDataGenerator(zoom_range=0.5))
    train_datagen.append(ImageDataGenerator(brightness_range=[0.2,1]))
    train_datagen.append(ImageDataGenerator(horizontal_flip=True))
    train_datagen.append(ImageDataGenerator(height_shift_range=0.5))
    train_datagen.append(ImageDataGenerator(width_shift_range=0.5))

    for gen in train_datagen :
        train_generator = gen.flow(
                image, 
                label,
                batch_size=len(image),
                shuffle=True)

        tmp_image,tmp_label  = next(train_generator)
        new_image = np.concatenate((new_image,tmp_image),axis=0)
        new_label = np.concatenate((new_label,tmp_label),axis=0)

    return new_image,new_label

