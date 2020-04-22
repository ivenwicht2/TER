from PIL import Image
from scipy import spatial
import torch 
import numpy as np 
import os
import pickle

# Définition de la fonction de découpage

def crop(image, coord):
    """
    @param image_path: Chemin vers l'image à découper
    @param coords: Tuple de coordonnées x/y (x1, y1, x2, y2)
    @param saved_location: Chemin ou il faut enregistrer l'image
    """
    img_cropped = []
    for element in coord : 
        cropped_image = image.crop(element)
        img_cropped.append(cropped_image)

    return img_cropped


def extraction_rep(image):
    """ extrait l'espace de représentation """
    model = torch.load("save/model")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    simi = None
    def hook(module, input_, output):
        nonlocal simi
        simi = output
    model.classifier[1].register_forward_hook(hook)
    output = model.forward(input)
    return simi.cpu().detach().numpy()

def simi(image,nb=6):
    """ renvoie le path des images similaires """
    representation_total = pickle.load(open("save/representation.npy", 'rb'))
    representation_image,_ = extraction_rep(image)
    _,index = representation_total[0].query(representation_image,k=nb)
    path = []
    for el in index :
        path.append(representation_total[1][el])
    return path 

def stack_rep(path):
    """ sauvegarde l'espace de représentation """
    representation = [[],[]]
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                image = Image.open(os.path.join(r, file))/255
                rep = extraction_rep(image)
                representation[0].append(rep)
                representation[1].append(os.path.join(r, file))
    pickle.dump(representation, open('save/representation.npy', 'wb'))