import os
from PIL import Image
import numpy as np

def loading_histo(list_path):
    list_img = []
    list_label = []
    for path in list_path :
        if os.path.isfile('project/active_learning/'+path) :
            #img = Image.open('project/active_learning/'+path)
            list_img.append('project/active_learning/'+path)

        elif os.path.isfile('project/stock_image/'+path) :
            #img = Image.open('project/stock_image/'+path)
            list_img.append('project/stock_image/'+path)


        else :
            tmp_path = path.split('/')[1]
            list_img.append('project/images/'+tmp_path)
        
        label = path.split('/')[0]
        list_label.append(label)
    
    print(np.shape(list_img),np.shape(list_label))
    return list_img, list_label