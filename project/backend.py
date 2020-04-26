from scipy import spatial
from PIL import Image
import numpy as np
import os
import torch 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from .active_learning import sorting 

def model_prediction(url,device,model,all_simi,all_path):

    
    url = url.split(r'/')[-1]
    url = 'project/images/' + url
    im = Image.open(url)
    im = im.convert('RGB')
    im = im.resize((224,224), Image.ANTIALIAS)
    im = TF.to_tensor(im).to(device)
    im.unsqueeze_(0)
    
    output = model.forward(im)
    result = F.softmax(output,dim=1)

    prediction = result.cpu().detach().numpy()
    simi = output.cpu().detach().numpy()

    _,index=all_simi.query(simi,k=5)
    save_img = []
    label_simi = []
    for el in index[0] :
        tmp_path = all_path[el]
        tmp_img = Image.open("project/stock_image/"+tmp_path)
        label_simi.append(   tmp_path.split('/')[0]      )
        save_img.append(  np.array(tmp_img)   )

    classes = os.listdir( os.path.realpath("project/stock_image"))


    sorting(url,prediction,classes[np.argmax(prediction)])


    return save_img,label_simi,[classes[np.argmax(prediction)]]

