from torchvision import models
from image_module import *
from torch import nn , optim 
import torch
from torch.utils.data import DataLoader
import numpy as np

class Convnet :
    def __init__(self,classe):
        print("création du modèle : ",end='')
        self.classe = classe
        device = torch.device("cpu")
        self.model = models.vgg19(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.ftrs = self.model.classifier[6].in_features
        features = list(self.model.classifier.children())[:-1]
        features.extend([torch.nn.Linear(self.ftrs, classe)])
        self.model.classifier = torch.nn.Sequential(*features)
        self.model = self.model.to(device)
        self.criteration = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam( self.model.parameters())
        print('done')
    def train(self,n_epochs,train,test):
        print("training")
        train_loader = train
        test_loader = test
        for epoch in range(n_epochs):
            print("epoch {} : ".format(epoch),end='')
            val_loss_train = []
            val_loss_test = []
            true_result = 0
            total = 0
            for (data_train, targets_train) , (data_test , targets_test) in zip(train_loader,test_loader):
               
                # Train
                out = self.model(data_train)
                loss = self.criteration(out, targets_train)
                loss.backward()
                self.optimizer.step()
                val_loss_train.append(loss.item())

                # Test 
                out = self.model(data_test)
                loss = self.criteration(out, targets_test)
                val_loss_test.append(loss.item())
                for output,label in zip(out.detach().numpy(), targets_test.detach().numpy()):
                    if np.argmax(output) == label : true_result  += 1
                    total += 1

                print('-',end='')
            acc = true_result / total

            val_loss_train =  np.average(val_loss_train)
            val_loss_test =  np.average(val_loss_test)
            print('> training loss: ', val_loss_train,' test loss: ',val_loss_test,' acc_test : ',acc)


if __name__ == '__main__' :

    test = Convnet(2)
    #print(test.model)
    traind,testd = import_img("DATA")
    test.train(10,testd,traind)
