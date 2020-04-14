import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import numpy as np 
import torch 
from torch import nn 
from torchvision import models

def detection(num_c):

    backbone = models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    #backbone = torch.load("model").features 
    #backbone.out_channels =  1280
    
    #backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-4])
    model = FasterRCNN(backbone,num_classes=num_c)
    
    return model



def get_prediction(model,img_path, threshold,Class):
    img = Image.open(img_path) # Load the image
    img = np.asarray(img)
    transform = torchvision.T.Compose([torchvision.T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    pred_class = [Class[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class
