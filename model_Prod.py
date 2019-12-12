from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Sequential, Model
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from scipy import spatial
import pathlib
from sklearn.model_selection import train_test_split
import pickle

data_dir = pathlib.Path("DATA")
image_count = len(list(data_dir.glob('*/*')))
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # redimension des images
BATCH_SIZE = image_count
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))
#On importe les images
with open(r"Sauvegarde modèle/path.txt", "wb") as fp:  #sauvegarde de l'emplacement des fichiers
     pickle.dump(train_data_gen.filenames, fp)

def augm(datagen):
    """
    Fonction qui renvoie une liste d'images modifier.
    Entrée : Mofication de type keras.preprocessing.image.ImageDataGenerator.
    Sortie : Images modifier sous forme de liste numpy.
    """
    datagen.fit(image_batch)
    augmT = []
    for el in datagen.flow(image_batch,shuffle=False,batch_size=BATCH_SIZE):
        for i in range(0, len(el)):
            if i == 0 : augmT.append(el[i])
            else : augmT.append(el[i])

        break
    augmT =  np.array(augmT)
    return augmT

datagen = []
datagen.append(ImageDataGenerator(rotation_range=30))
datagen.append(ImageDataGenerator(zoom_range=[0.5,1.0]))
datagen.append(ImageDataGenerator(brightness_range=[0.9,1.01]))
datagen.append(ImageDataGenerator(fill_mode='constant'))
datagen.append(ImageDataGenerator(cval=255))
datagen.append(ImageDataGenerator(width_shift_range=[-50,50]))


for i,gen in enumerate(datagen) :
    if i == 0:
        new_img =  np.concatenate((image_batch,augm(gen)))
        new_label =  np.concatenate((label_batch,label_batch))
    else: 
        new_img =  np.concatenate((new_img,augm(gen)))
        new_label =  np.concatenate((new_label,label_batch))

(trainX, testX, trainY, testY) = train_test_split(new_img, new_label, test_size=0.25)

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (224, 224, 3)) # transfer learning
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(len(CLASS_NAMES), activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
history = model_final.fit(trainX, trainY, epochs=10, 
                    validation_data=(testX, testY))

model_final.save("Sauvegarde modèle/model_sauvegarde")

model_simi = Model(input = model.input, output =x)

pred = model_simi.predict(image_batch) # prédiction sur toutes les images non modifié 

model_simi.save("Sauvegarde modèle/simi_sauvegarde")
np.save('Sauvegarde modèle/representation.npy',  pred)
np.save('Sauvegarde modèle/img.npy', image_batch )
np.save('Sauvegarde modèle/label.npy', label_batch )
np.save('Sauvegarde modèle/class.npy', CLASS_NAMES )

with open(r"Sauvegarde modèle/result.txt", "wb") as fp:  #sauvegarde de l'emplacement des fichiers
     pickle.dump(test_acc, fp)
