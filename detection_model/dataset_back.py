import os
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np 

dic = { 
    'D4' : '1000000000',
    'D21' : '0100000000',
    'F35' : '0010000000',
    'G17' : '0001000000',
    'G43' : '0000100000',
    'I10' : '0000010000',
    'M17' : '0000001000',
    'N35' : '0000000100',
    'V30' : '0000000010',
    'X8' : '0000000001'
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
				boxes.append(box)
				label =  {e.tag: e.text for e in root.findall('.//name')}
				label_encod = dic[label['name']]
				labels.append(label_encod)

			if "filename" in child.tag :
				#Image
				image_target = child.text
				img = Image.open( path+"/JPEGImages/"+image_target)
				Image_total.append( np.asarray(img) ) 

	target = {}
	target["boxes"] = boxes
	target["labels"] = labels
	data.append(target)

	return data , Image_total



if __name__ == "__main__":
    import_dataset("C:/Users/theoo/OneDrive/Documents/ter/test/VOCdevkit")
