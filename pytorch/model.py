from torchvision import models
from image_module import *
from torch import nn , optim 
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from train import *


def Convnet(model_type,classe) :
        print("création du modèle : ",end='')

        print(' ',model_type)

        if model_type == 'vgg16' :
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
                in_features = 25088
        if model_type == 'densenet121' :
            model = models.densenet121(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
                in_features= 1024

        fc = nn.Sequential(
            nn.Linear(in_features, 460),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(460,classe),  
        )
        model.classifier = fc
        print('done')
        return model    

    
if __name__ == '__main__' :

    test = Convnet('vgg16',3)
    trainloader,testloader = import_img("DATA")
    test = train_model(test,trainloader,testloader,10)