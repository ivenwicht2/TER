from torch import nn , optim 
import torch
import numpy as np
import time

def train_model(model,train_loader,valid_loader,epochs,lr=0.01):
        print("cuda" if torch.cuda.is_available() else "cpu")
        print("training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss().cuda()
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
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
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
