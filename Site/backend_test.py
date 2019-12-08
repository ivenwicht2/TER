from keras.models import load_model
import tensorflow as tf
from scipy import spatial
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle

im = Image.open(r"C:\Users\theoo\OneDrive\Documents\GitHub\TER\Site\images\KIU1034Poursvers_4.png")
im = im.convert('RGB')
im = im.resize((224,224), Image.ANTIALIAS)
im = np.array(im)
im = np.expand_dims(im, axis=0)

model = load_model(r"script\save\model_sauvegarde")
representation = load_model(r"script\save\simi_sauvegarde")
img = np.load(r"script\save\img.npy")
label = np.load(r"script\save\label.npy")
Class = np.load(r"script\save\class.npy")
simi = np.load(r"script\save\representation.npy")

pred = model.predict(im)
pred = pred.argmax(axis=1)[0]
print(pred)

"""
def similarite(nb,quer):
    distance,index = spatial.KDTree(simi).query(quer,k=nb+1)  
    plt.imshow(np.squeeze(img[index[0]]))
    distance = distance[1:]
    index = index[1:]
    #plt.show()
    plt.figure(figsize=(10,10))
    for n in range(len(index)):
          ax = plt.subplot(5,5,n+1)
          plt.imshow(np.squeeze(img[index[n]]))
          plt.title(Class[label[index[n]]==1][0].title())
          plt.axis('off')
    plt.show()
"""
quer = representation.predict(im)[0]
nb = 6
#similarite(nb,quer)

distance,index = spatial.KDTree(simi).query(quer,k=nb+1)  
print("test")
with open(r"script\save\path.txt", "rb") as fp:   # Unpickling
     path_total = pickle.load(fp)
path = []
print('path : ',path_total)
for i in range(len(index)): 
    path.append(path_total[index[i]])
print(path)
