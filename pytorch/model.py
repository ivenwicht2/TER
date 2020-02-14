from torchvision import models
from image_module import *
from torch import nn 
import torch
from torch.utils.data import DataLoader
import numpy as np
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

        if model_type == 'vgg19' :
            model = models.vgg19(pretrained=True)
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
        return model    

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def pred(model,input) :
    support = "cuda" if torch.cuda.is_available() else "cpu"
    if support == "cuda":
        device = torch.device("cuda")
    else : 
        device = torch.device("cpu")

    print("pred on ",end='')
    print("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = input.to(device)
    output = model.forward(input)
    print(output.cpu().detach().numpy())
    proba = softmax(output.cpu().detach().numpy())
    predicted = np.argmax(proba)
    
    print(predicted,proba)

if __name__ == '__main__' :

    test = Convnet('vgg16',3)
    trainloader,testloader = import_img("DATA")
    test = train_model(test,trainloader,testloader,3)

    from PIL import Image
    import torchvision.transforms.functional as TF

    image = Image.open('DATA/G43/G43_KIU862_55362 2.jpg')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    pred(test,x)