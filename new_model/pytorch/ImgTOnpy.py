from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt

def extract(path):
        """Extrait les images d'un dossier
        Input = chemin du dossier contenant toutes les images
        Output  = 
            liste numpy de toutes les images en format (224,224) normalise,
            Label de toutes les images correspondant au nom du dossier qui les contients,
        """
        data_dir = pathlib.Path(path)
        #image_count = len(list(data_dir.glob('*/*'))) nombre d'images extrait
        image_list = []
        Class = []
        for r, _, f in os.walk(data_dir):
            for file in f:
                im=imread (os.path.join(r, file))
                im = resize(im,(224,224))
                im = np.array(im)
                im = np.divide(im,255)
                im = im.astype('float32')
                image_list.append(im)
                Class.append(r.split("\\")[1])
        return image_list,Class

def show_img(img_list,title = None,nb=4):
    """Affiche les images en format numpy et normalise 
    Input = 
        Liste contenant des images en format numpy
        Nom de chaque images (typiquement leur label)
        Nombre d'image à afficher 
    Output = Affiche les images
    """
    fig = plt.figure()
    for i in range(nb):
        ax = fig.add_subplot(1, nb, i+1)
        ax.imshow(img_list[i]*255, interpolation='nearest')
        ax.set_title(title[i])

    plt.show()
if __name__ == '__main__':
    img,label = extract("DATA")
    show_img(img,label,5)
 

