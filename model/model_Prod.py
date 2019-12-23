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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

data_dir = pathlib.Path("DATA")
image_count = len(list(data_dir.glob('*/*')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]) 
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
image_batch, label_batch = next(train_data_gen)
def augm(datagen,img_dt):
    """
    Fonction qui renvoie une liste d'images modifier.
    Entree : Mofication de type keras.preprocessing.image.ImageDataGenerator.
    Sortie : Images modifier sous forme de liste numpy.
    """
    datagen.fit(img_dt)
    augmT = []
    for el in datagen.flow(img_dt,shuffle=False,batch_size=BATCH_SIZE):
        for i in range(0, len(el)):
            if i == 0 : augmT.append(el[i])
            else : augmT.append(el[i])

        break
    augmT =  np.array(augmT)
    return augmT


(trainX, testX, trainY, testY) = train_test_split(image_batch, label_batch, test_size=0.25) 

datagen = []
datagen.append(ImageDataGenerator(rotation_range=30))
datagen.append(ImageDataGenerator(zoom_range=[0.5,1.0]))
datagen.append(ImageDataGenerator(brightness_range=[0.2,1.0]))
datagen.append(ImageDataGenerator(horizontal_flip=True))
datagen.append(ImageDataGenerator(width_shift_range=[-50,50]))
datagen.append(ImageDataGenerator(height_shift_range=0.5))


for i,gen in enumerate(datagen) :
    if i == 0:
        new_testX =  np.concatenate((testX,augm(gen,testX)))
        new_trainX =  np.concatenate((trainX,augm(gen,trainX)))
        new_testY =  np.concatenate((testY,testY))
        new_trainY =  np.concatenate((trainY,trainY))
    else: 
        new_testX =  np.concatenate((new_testX,augm(gen,testX)))
        new_trainX =  np.concatenate((new_trainX,augm(gen,trainX)))
        new_testY =  np.concatenate((new_testY,testY))
        new_trainY =  np.concatenate((new_trainY,trainY))


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (224, 224, 3)) # transfer learning
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(len(CLASS_NAMES), activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
history = model_final.fit(new_trainX,new_trainY, epochs=10, 
                    validation_data=(new_testX, new_testY))
metric_result = model_final.predict(new_testX, batch_size=len(new_testX), verbose=1)

predicted_classes = np.argmax(metric_result , axis=1)
test_matrice = np.argmax(new_testY, axis=1)

print(classification_report(test_matrice, predicted_classes, 
        target_names=CLASS_NAMES , digits = 6))

model_final.save("model_save/model_sauvegarde")
############# sauvegarde resultat epoch #####################
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('Epoch.png')
model_simi = Model(input = model.input, output =x)
###############################################################

model_simi = Model(input = model.input, output =x)

pred = model_simi.predict(image_batch) # prediction sur toutes les images non modifie 

model_simi.save("model_save/simi_sauvegarde")
np.save('model_save/representation.npy',  pred)
np.save('model_save/img.npy', image_batch )
np.save('model_save/label.npy', label_batch )
np.save('model_save/class.npy', CLASS_NAMES )

