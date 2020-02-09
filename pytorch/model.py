from torchvision import models
from image_module import *
from torch import nn , optim 
import torch
from torch.utils.data import DataLoader
import numpy as np
import time


class Convnet :
    def __init__(self,classe):
        print("création du modèle : ",end='')
        self.classe = classe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.densenet121(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        fc = nn.Sequential(
            nn.Linear(1024, 460),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(460,2),
            nn.LogSoftmax(dim=1)   
        )

        self.model.classifier = fc

        self.criterion = nn.NLLLoss()
        self. optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=0.003)
        self.model.to(self.device)

        print('done')

    def train(self,epochs,train_loader,valid_loader):
        print("training")

        valid_loss_min = np.Inf
        for epoch in range(epochs):
    
            start = time.time()
    
            #scheduler.step()
            self.model.train()
    
            train_loss = 0.0
            valid_loss = 0.0
    
            for inputs, labels in train_loader:
        
       
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
        
                self.optimizer.zero_grad()
            
                logps = self.model(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
        
       
            self.model.eval()
    
            with torch.no_grad():
                accuracy = 0
                for inputs, labels in valid_loader:
            
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        logps = self.model.forward(inputs)
                        batch_loss = self.criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
            # calculate average losses
            train_loss = train_loss/len(train_loader)
            valid_loss = valid_loss/len(valid_loader)
            valid_accuracy = accuracy/len(valid_loader) 
      
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
                epoch + 1, train_loss, valid_loss, valid_accuracy))
            
            """if valid_loss <= valid_loss_min:      
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                model_save_name = "test"
                path = F"/content/drive/My Drive/{model_save_name}"
                torch.save(model.state_dict(), path)
                valid_loss_min = valid_loss 
            """      

            print(f"Time per epoch: {(time.time() - start):.3f} seconds")


if __name__ == '__main__' :

    test = Convnet(2)
    #print(test.model)
    trainloader,testloader = import_img("DATA")
    test.train(100,trainloader,testloader)
