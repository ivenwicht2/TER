import os
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np 
import torch
import transforms as T
from torchvision import transforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class PennFudanDataset(object):
	def __init__(self, root,transforms):
		self.root = root
		self.transforms = transforms
		# load all image files, sorting them to
		# ensure that they are aligned
		self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
		self.annot = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
		self.dict = { 
	            'D4'  : 0,
	            'D21' : 1,
	            'F35' : 2,
	            'G17' : 3,
	            'G43' : 4,
	            'I10' : 5,
	            'M17' : 6,
	            'N35' : 7,
	            'V30' : 8,
        	    'X8'  : 9 
	        }

	def __getitem__(self, idx):
		# load images ad masks
		img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
		annotation = os.path.join(self.root, "Annotations", self.annot[idx])

		img = Image.open(img_path).convert("RGB")

		boxes = []
		root = ET.parse(annotation).getroot()
		labels = []
		num_objs = 0
		area = []
		for child in root : 
			if "object" in child.tag :
				box = {e.tag: int(e.text) for e in child.findall('.//bndbox/*')}
				tmp = [box['xmin'],box['ymin'],box['xmax'],box['ymax']]
				boxes.append(tmp)
				label =  {e.tag: e.text for e in child.findall('.//name')}
				label_encod = self.dict[label['name']]
				labels.append(label_encod)
				area.append((tmp[3] - tmp[1]) * (tmp[2] - tmp[0]))
				num_objs += 1

		# convert everything into a torch.Tensor
		image_id = torch.tensor([idx])
		boxes = torch.as_tensor(boxes, dtype=torch.float32).to(device)
		labels = torch.as_tensor(np.array(labels), dtype=torch.int64).to(device)
		target = {}
		target['image_id'] = image_id
		target["boxes"] = boxes
		target["labels"] = labels
		target["area"] = torch.as_tensor(area)
		target["iscrowd"] =  torch.zeros((num_objs,), dtype=torch.int64)
		if self.transforms is not None :
			img, target = self.transforms(img, target)
		return img.to(device), target

	def __len__(self):
       		return len(self.imgs)