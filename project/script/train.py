from torch import nn , optim 
import torch
import numpy as np
import time
import os 



def train_model(model,train_loader,valid_loader,epochs,lr=0.01,save=0):
        print("support : ",end='')
        support = "cuda" if torch.cuda.is_available() else "cpu"
        print(support)
        print("training")
        if support == "cuda":
            device = torch.device("cuda")
            criterion = nn.CrossEntropyLoss().cuda()
        else : 
            device = torch.device("cpu")
            criterion = nn.CrossEntropyLoss().cpu()

            
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
        model.to(device)
        valid_loss_min = np.Inf
        for epoch in range(epochs):
    
            start = time.time()
    
            #scheduler.step()
            model.train()
    
            train_loss = 0.0
            valid_loss = 0.0
    
            for inputs, labels in train_loader:
       
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
        
                optimizer.zero_grad()
            
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
       
            model.eval()
    
            with torch.no_grad():
                accuracy = 0
                for inputs, labels in valid_loader:
            
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        valid_loss += batch_loss.item()
                        # Calculate accuracy
                        top_p, top_class = output.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
            # calculate average losses
            train_loss = train_loss/len(train_loader)
            valid_loss = valid_loss/len(valid_loader)
            valid_accuracy = accuracy/len(valid_loader) 
      
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
                epoch + 1, train_loss, valid_loss, valid_accuracy))
            
            if valid_loss <= valid_loss_min and save == 1:      
                model_save_name = "model"
                path = f"save/{model_save_name}"
                #torch.save(model.state_dict(), path)
                torch.save(model, path)
                valid_loss_min = valid_loss 

            print(f"Time per epoch: {(time.time() - start):.3f} seconds")
        return model
