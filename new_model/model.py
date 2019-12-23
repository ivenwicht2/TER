from augmentation import augment
from importation import extract
import tensorflow as tf 
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras import applications
from keras.layers import Dense,Flatten, Dropout
from keras.models import Model

from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"

img,label,Class = extract("DATA")
(trainX, testX, trainY, testY) = train_test_split(img, label, test_size=0.25)


new_trainX,new_trainY = augment(trainX,trainY)
new_testX,new_testY = augment(testX,testY)

base = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (224, 224, 3)) # transfer learning
#base = ResNet50(include_top=False, weights='imagenet',input_shape = (224, 224, 3))
x=base.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
preds=Dense(len(Class),activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base.input,outputs=preds)



model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
   
history = model.fit(new_trainX, new_trainY, epochs=10,validation_data=(new_testX, new_testY))

plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label = 'val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()