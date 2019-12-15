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
    img = np.load("script/save/img.npy")
    Class = np.load("script/save/class.npy")
    simi = np.load("script/save/representation.npy")
    pred = model.predict(im)
    pred = pred.argmax(axis=1)[0]
    quer = representation.predict(im)[0]
    nb = 6
    save_img = []
    _,index = spatial.KDTree(simi).query(quer,k=nb)
    with open("script/save/path.txt", "rb") as fp:  
        path_total = pickle.load(fp)
    path = []
    for i in range(len(index)):
        tmp = "/static/" + path_total[index[i]].replace("\\","/") 
        path.append(tmp)
        save_img.append(img[index[i]])
    return save_img,Class[pred]

