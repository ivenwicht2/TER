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
    print("this is the url : ",url)
    im = Image.open(url)
    im = im.convert('RGB')
    im = im.resize((224,224), Image.ANTIALIAS)
    im = np.array(im)
    print(np.shape(im))
    im = np.expand_dims(im, axis=0)
    print("oui1")
    model = load_model(r"script\save\model_sauvegarde")
    print("oui2")
    representation = load_model(r"script\save\simi_sauvegarde")
    print("oui3")
    img = np.load(r"script\save\img.npy")
    print("oui4")
    label = np.load(r"script\save\label.npy")
    print("oui5")
    Class = np.load(r"script\save\class.npy")
    print("oui6")
    simi = np.load(r"script\save\representation.npy")
    print("oui7")
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
    print("script fini")
    return path,Class[pred]

