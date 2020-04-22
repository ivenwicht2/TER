from scipy import spatial
from PIL import Image
import numpy as np
import pickle
import os

def model(url):
    #folder_path = "/Users/priscille/Desktop/Site/images"
    #save_img=[]
    #for files in os.walk(folder_path):
    #    for img in files:
    #        save_img.append(img)
    cwd = os.getcwd()
    save_img=Image.open("/Users/priscille/Documents/GitHub/TER/Site/images/Oeil02_2.jpg")
    save_img=np.array(save_img)
    print(cwd)
    label_simi="D21" #images similaires
    Class="D21" #images en input
    return [save_img],[label_simi],[Class]


