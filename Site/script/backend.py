from keras.models import load_model
import tensorflow as tf
from scipy import spatial
from PIL import Image
import numpy as np
import pickle
import os

def model(url):
    tmp = os.path.realpath('images')
    url = url.split(r'/')[-1]
    url = tmp + "\\" + url
    im = Image.open(url)
    im = im.convert('RGB')
    im = im.resize((224,224), Image.ANTIALIAS)
    im = np.array(im)
    im = np.expand_dims(im, axis=0)
    model = load_model("script/save/model_sauvegarde",)
    representation = load_model("script/save/simi_sauvegarde")
    label= np.load("script/save/label.npy")
    img = np.load("script/save/img.npy")
    Class = np.load("script/save/class.npy")
    simi = np.load("script/save/representation.npy")
    pred = model.predict(im)
    pred = pred.argmax(axis=1)[0]
    quer = representation.predict(im)[0]
    nb = 6
    save_img = []
    _,index = spatial.KDTree(simi).query(quer,k=nb)
    label_simi = []
    for i in range(len(index)):
        tmp = np.expand_dims( label[index[i]], axis=0)
        tmp =  tmp.argmax(axis=1)[0]
        label_simi.append(Class[tmp])
        save_img.append(img[index[i]])
    return save_img,label_simi,Class[pred]

