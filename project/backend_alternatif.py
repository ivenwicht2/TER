
import torchvision.transforms.functional as TF
import torch
from PIL import Image,ImageDraw
import numpy as np 

def _drawBoundingBox(img,result):
        tmp = np.array(result[0]["boxes"].cpu().detach())
        tmp2 = np.array(result[0]['labels'].cpu().detach())
        tmp3 = np.array(result[0]['scores'].cpu().detach())
        for el,lab,score in zip(tmp,tmp2,tmp3) :
                if score > 0:
                        x1,y1,x2,y2 = el
                        print(x1,y1,x2,y2)
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([x1,y1,x2,y2])
                        draw.text((x1,y1),str(lab), fill=(255,255,255,128))
                        del draw

        return img

def prediction(url):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_clean = Image.open(url)
        img = TF.to_tensor(img_clean).to(device)
        model = torch.load("script/save/detection_model")
        model.eval()
        result = model([img])
        new_img = _drawBoundingBox(img_clean,result)
        return new_img 
