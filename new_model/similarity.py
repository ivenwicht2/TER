from keras.models import load_model
from scipy import spatial
import tensorflow as tf
import numpy as np 

model = load_model("save/model",)
representation = load_model("save/simi")
label= np.load("save/label.npy")
img = np.load("save/img.npy")
Class = np.load("save/class.npy")
representation = np.load("save/representation.npy")


quer = representation[0]
nb = 6
_,index = spatial.KDTree(representation).query(quer,k=nb)
label_simi = []
for i in range(len(index)):
    tmp = np.expand_dims( label[index[i]], axis=0)
    tmp =  tmp.argmax(axis=1)[0]
    label_simi.append(Class[tmp])
    save_img.append(img[index[i]])
