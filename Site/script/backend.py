from keras.models import load_model
import tensorflow as tf
from scipy import spatial
from PIL import Image
import numpy as np
import pickle
import os


def model(url):
    print("path is equal to ",os.getcwd())
    tmp = "C:\\Users\\theoo\\OneDrive\\Documents\\GitHub\\TER\Site\\images\\"
    url = url.split(r'/')[-1]
    url = tmp + url
    im = Image.open(url)
    im = im.convert('RGB')
    im = im.resize((224,224), Image.ANTIALIAS)
    im = np.array(im)
    im = np.expand_dims(im, axis=0)
    model = load_model(r"script\save\model_sauvegarde",)
    representation = load_model(r"script\save\simi_sauvegarde")
    img = np.load(r"script\save\img.npy")
    label = np.load(r"script\save\label.npy")
    Class = np.load(r"script\save\class.npy")
    simi = np.load(r"script\save\representation.npy")
    pred = model.predict(im)
    pred = pred.argmax(axis=1)[0]
    quer = representation.predict(im)[0]
    nb = 6
    distance,index = spatial.KDTree(simi).query(quer,k=nb+1)
    with open(r"script\save\path.txt", "rb") as fp:   # Unpickling
        path_total = pickle.load(fp)
    path = []
    for i in range(len(index)): 
        path.append(path_total[index[i]])
        print(path_total[index[i]])
    return path,Class[pred]

