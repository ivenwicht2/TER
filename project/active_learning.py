import numpy as np
import shutil
import os 
from PIL import Image

def sorting(path,proba,classe):

    entropy = 0
    pathwclass = classe + '/' + path.split('/')[-1]

    for p in proba[0] :
        entropy -= p* np.log(p)
    
    if entropy > 0.75*np.log(10):
        new_path = "project/stock_image/" + pathwclass
    else :
        new_path = "project/active_learning/" + pathwclass
        if  os.path.isfile('project/script/save/entropy.npy'):
            rank = np.load('project/script/save/entropy.npy')
            if len(rank) == 0 :
                rank = np.array([[entropy,new_path]])
            else : 
                rank = np.vstack([rank,[entropy,new_path]])
        else : 
            rank = np.array([[entropy,new_path]])

        np.save('project/script/save/entropy.npy',rank)

    shutil.copy(path, new_path)



def learning():
    if os.path.isfile('project/script/save/entropy.npy'):
        rank = np.load('project/script/save/entropy.npy')
        rank =  np.sort(rank,axis=0)
        print("raaaaaaaaaannnnnnkkkkkkk",rank)
        if len(rank) == 0 :
            return None
        elif len(rank) == 2 and rank.ndim == 1 : 
            print("////////////////",rank)
            path  = rank[1]
        else : 
            minimum = rank[0]
            path = minimum[1]
 
        img = Image.open(path).convert('RGB')
        return img
    else :
        return None



def move_img(classe):
    rank = np.load('project/script/save/entropy.npy')
    rank =  np.sort(rank,axis=0)
    if len(rank) == 2 and rank.ndim == 1 : 
            path  = rank[1]
            minimum = rank
    else : 
        minimum = rank[0]
        path = minimum[1]


    new_path = "project/stock_image/"+classe+"/"+ path.split('/')[-1]
    print("iiiiiiiiiiiiiiiii",path,new_path)
    shutil.copy(path, new_path)

    print("**********************",rank )
    
    rank = rank[rank !=minimum]

    print("##################",minimum , rank)

    np.save('project/script/save/entropy.npy',rank)
