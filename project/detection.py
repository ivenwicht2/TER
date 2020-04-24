from PIL import Image,ImageDraw,ImageFont
import os
import torchvision.transforms.functional as TF
import torch
import numpy as np

def crop(im,height,width):
    imgwidth, imgheight = im.size
    integers = int(imgheight//height)
    for i in range(integers):
        for j in range(integers):
            # print (i,j)
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)

def _drawBoundingBox(img,result):
        classes = os.listdir( os.path.realpath("project/stock_image"))
        tmp = np.array(result[0]["boxes"].cpu().detach())
        tmp2 = np.array(result[0]['labels'].cpu().detach())
        tmp3 = np.array(result[0]['scores'].cpu().detach())
        for el,lab,score in zip(tmp,tmp2,tmp3) :
                if score > 0.80:
                        x1,y1,x2,y2 = el
                        draw = ImageDraw.Draw(img)
                        #draw.rectangle([x1,y1,x2,y2])
                        line = (x1,y1,x1,y2)
                        draw.line(line, fill="red", width=10)
                        line = (x1,y1,x2,y1)
                        draw.line(line, fill="red", width=10)
                        line = (x1,y2,x2,y2)
                        draw.line(line, fill="red", width=10)
                        line = (x2,y1,x2,y2)
                        draw.line(line, fill="red", width=10)
                        classe = classes[lab]
                        font = ImageFont.truetype("arial.ttf", 50)
                        draw.text((x1-50,y1),str(classe),font=font, fill=(255,255,255,128))
                        del draw

        return img

def detection_image(infile,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    filenum = 1
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    height = imgheight/4
    width =  imgwidth/4
    start_num = 0
    new_img = []
    for k,piece in enumerate(crop(im,height,width),start_num):

        img=Image.new('RGB', (int(width),int(height)), 255)
        img.paste(piece)
        img_tensor = TF.to_tensor(img).to(device)
        result = model([img_tensor])
        new_img.append(_drawBoundingBox(img,result))
    
    size = new_img[0].size
    tmp_img = Image.new('RGB', (4*size[0],4*size[1]), (250,250,250))


    tmp_img.paste(new_img[0], (0,0))
    tmp_img.paste(new_img[1], (size[0],0))
    tmp_img.paste(new_img[2], (2*size[0],0))
    tmp_img.paste(new_img[3], (3*size[0],0))

    tmp_img.paste(new_img[4], (0,size[1]))
    tmp_img.paste(new_img[5], (size[0],size[1]))
    tmp_img.paste(new_img[6], (2*size[0],size[1]))
    tmp_img.paste(new_img[7], (3*size[0],size[1]))
    
    return tmp_img