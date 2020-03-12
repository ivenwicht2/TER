import os
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np 

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
                boxes.append(box)
                label =  {e.tag: e.text for e in root.findall('.//name')}
                labels.append(label['name'])
        
            if "filename" in child.tag :
                #Image
                image_target = child.text
                img = Image.open( path+"/JPEGImages/"+image_target)
                Image_total.append( np.asarray(img) ) 
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        data.append(target)

    print(Image_total)
    return data , Image_total



if __name__ == "__main__":
    import_dataset("C:/Users/theoo/OneDrive/Documents/ter/test/VOCdevkit")
