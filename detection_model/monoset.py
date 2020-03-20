import torch 
from PIL import Image
import xml.etree.ElementTree as ET
import os
import numpy as np

dic = { 
    'D4' : [1,0,0,0,0,0,0,0,0,0],
    'D21' : [0,1,0,0,0,0,0,0,0,0],
    'F35' : [0,0,1,0,0,0,0,0,0,0],
    'G17' : [0,0,0,1,0,0,0,0,0,0],
    'G43' : [0,0,0,0,1,0,0,0,0,0],
    'I10' : [0,0,0,0,0,1,0,0,0,0],
    'M17' : [0,0,0,0,0,0,1,0,0,0],
    'N35' : [0,0,0,0,0,0,0,1,0,0],
    'V30' : [0,0,0,0,0,0,0,0,1,0],
    'X8' : [0,0,0,0,0,0,0,0,0,1]
}

def import_dataset(path):
    liste = filter(lambda x: x.endswith('.xml'), os.listdir(os.path.join(path,"Annotations")))
    data = []
    for file in liste :
        #xml
        root = ET.parse( os.path.join(path,"Annotations/"+file) ).getroot()
        boxes = []
        labels = []
        Image_total = []
        for child in root :
            if "object" in child.tag :
                box = {e.tag: int(e.text) for e in root.findall('.//bndbox/*')}
                tmp = [box['xmin'],box['ymin'],box['xmax'],box['ymax']]
                boxes.append(torch.from_numpy(np.array(tmp)))
                label =  {e.tag: e.text for e in root.findall('.//name')}
                label_encod = dic[label['name']]
                labels.append(torch.from_numpy(np.array(label_encod)))

            if "filename" in child.tag :
                #Image
                image_target = child.text
                img = Image.open( path+"/JPEGImages/"+image_target).convert('RGB')
                tmp = torch.from_numpy(np.asarray(img))
                #print(np.shape(tmp))
                Image_total.append(tmp) 

        target = {}

        target["boxes"] = boxes
        target["labels"] = labels
        data.append(target)
    Image_total = torch.stack(Image_total).float() 
    return Image_total, data
